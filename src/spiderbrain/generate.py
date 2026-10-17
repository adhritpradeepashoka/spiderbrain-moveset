"""Sample one synthetic connectome from the priors in configs/."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .provenance import make_provenance

Z90 = 1.645


@dataclass
class Connectome:
    seed: int
    neurons: pd.DataFrame            # one row per neuron
    edges: pd.DataFrame              # src, dst, syn (signed count), proj
    region_names: list[str]
    proj_names: list[str]
    provenance: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.neurons)

    def weights(self) -> sp.csr_matrix:
        """Signed synapse counts, rows = target, cols = source."""
        e = self.edges
        return sp.csr_matrix((e["syn"].to_numpy(np.float32), (e["dst"], e["src"])), shape=(self.n, self.n))

    def region_ids(self, region: str, hemi: int | None = None) -> np.ndarray:
        rid = self.region_names.index(region)
        m = self.neurons["region_id"].to_numpy() == rid
        if hemi is not None:
            m &= self.neurons["hemi"].to_numpy() == hemi
        return np.flatnonzero(m)


def _lognormal_in_range(rng, default, lo, hi):
    sigma = np.log(hi / lo) / (2 * Z90) if lo > 0 else 0.5
    v = default * np.exp(rng.normal(0, sigma))
    return float(np.clip(v, lo, hi))


def sample_region_sizes(cfg, rng, total) -> dict[str, int]:
    regs = cfg["regions"]["regions"]
    sample = cfg["ensemble"].get("sample_region_sizes", True)
    n = {}
    for r, s in regs.items():
        d = s["n"]["default"]
        lo, hi = s["n"]["range"]
        n[r] = _lognormal_in_range(rng, d, lo, hi) if sample else float(d)
    scal = [r for r, s in regs.items() if s.get("scalable", True)]
    fixed = sum(v for r, v in n.items() if r not in scal)
    scal_sum = sum(n[r] for r in scal)
    f = max(total - fixed, 0) / scal_sum
    out = {}
    for r in regs:
        v = n[r] * f if r in scal else n[r]
        out[r] = max(int(round(v)), 2 if regs[r].get("paired", True) else 1)
    return out


def _sample_fractions(cfg, rng, region):
    tx = cfg["transmitters"]
    fr = dict(tx.get("by_region", {}).get(region, tx["default"]))
    names = [k for k, v in fr.items() if v > 0]
    p = np.array([fr[k] for k in names], float)
    p /= p.sum()
    if len(names) > 1:
        p = rng.dirichlet(p * tx["dirichlet_concentration"])
    return names, p


def build_neurons(cfg, rng, sizes):
    regs = cfg["regions"]["regions"]
    tx = cfg["transmitters"]
    override = tx.get("sign_override", {})
    pexc = tx["transmitters"]["glutamate"].get("p_excitatory", 0.5)
    region_names = list(regs)
    frames = []
    tx_names = list(tx["transmitters"])
    for rid, r in enumerate(region_names):
        n = sizes[r]
        s = regs[r]
        if s.get("paired", True):
            hemi = (np.arange(n) % 2).astype(np.int8)
        else:
            hemi = np.full(n, 2, np.int8)
        pos = np.empty(n, np.float32)
        for h in np.unique(hemi):
            idx = np.flatnonzero(hemi == h)
            pos[idx] = (rng.permutation(len(idx)) + 0.5) / len(idx)
        layers = int(s.get("layers", 1))
        layer = rng.integers(0, layers, n).astype(np.int8) if layers > 1 else np.zeros(n, np.int8)
        names, p = _sample_fractions(cfg, rng, r)
        pick = rng.choice(len(names), size=n, p=p)
        tcode = np.array([tx_names.index(names[i]) for i in pick], np.int8)
        sign = np.ones(n, np.int8)
        for ti, tn in enumerate(tx_names):
            spec = tx["transmitters"][tn]
            m = tcode == ti
            if spec["sign"] == "sampled":
                sign[m] = np.where(rng.random(m.sum()) < pexc, 1, -1)
            else:
                sign[m] = int(spec["sign"])
        if r in override:
            sign[:] = int(override[r])
        modulatory = np.array([tx["transmitters"][tx_names[t]]["modulatory"] for t in tcode])
        pep = rng.random(n) < _peptide_frac(cfg, rng)
        frames.append(pd.DataFrame({
            "region_id": np.int16(rid), "hemi": hemi, "pos": pos, "layer": layer,
            "transmitter": tcode, "sign": sign, "modulatory": modulatory, "peptidergic": pep,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.index.name = "neuron_id"
    df["gamma_out"] = 0.0
    df["gamma_in"] = 0.0
    return df, region_names, tx_names


def _peptide_frac(cfg, rng):
    p = cfg["transmitters"]["peptidergic_cotransmission"]
    return _lognormal_in_range(rng, p["default"], *p["range"])


def _gamma(rng, shape, n):
    return rng.gamma(shape, 1.0 / shape, n)


def _wire(rng, proj, k, src, dst, df_src, df_dst, gsrc, gdst):
    """Return (src_ids, dst_ids) arrays, duplicates allowed."""
    ns = len(src)
    deg = rng.poisson(k * gsrc)
    s_rep = np.repeat(np.arange(ns), deg)
    if len(s_rep) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    s_hemi = df_src["hemi"].to_numpy()[s_rep]
    s_pos = df_src["pos"].to_numpy()[s_rep]
    s_layer = df_src["layer"].to_numpy()[s_rep]
    d_hemi = df_dst["hemi"].to_numpy()
    d_pos = df_dst["pos"].to_numpy()
    d_layer = df_dst["layer"].to_numpy()
    both_paired = (s_hemi != 2).all() and (d_hemi != 2).all()
    contra = float(proj.get("contra", 0.0))
    sigma = proj.get("retinotopic_sigma")
    matched = proj.get("layer_map") == "matched"
    if both_paired:
        flip = rng.random(len(s_rep)) < contra
        want_hemi = np.where(flip, 1 - s_hemi, s_hemi)
    else:
        want_hemi = np.full(len(s_rep), -1)
    out_t = np.empty(len(s_rep), np.int64)
    # group by (target hemisphere, layer) pools
    layer_key = s_layer if matched else np.zeros(len(s_rep), np.int8)
    for h in np.unique(want_hemi):
        for lay in np.unique(layer_key[want_hemi == h]):
            sel = np.flatnonzero((want_hemi == h) & (layer_key == lay))
            pool_mask = np.ones(len(d_hemi), bool)
            if h >= 0:
                pool_mask &= d_hemi == h
            if matched:
                pool_mask &= d_layer == lay
            pool = np.flatnonzero(pool_mask)
            if len(pool) == 0:
                pool = np.arange(len(d_hemi))
            w = gdst[pool]
            if sigma:
                order = np.argsort(d_pos[pool])
                pool, w = pool[order], w[order]
                target_pos = np.clip(s_pos[sel] + rng.normal(0, sigma, len(sel)), 0, 1 - 1e-6)
                ppos = d_pos[pool]
                i = np.searchsorted(ppos, target_pos)
                out_t[sel] = pool[np.clip(i, 0, len(pool) - 1)]
            else:
                c = np.cumsum(w)
                out_t[sel] = pool[np.searchsorted(c, rng.random(len(sel)) * c[-1])]
    return s_rep, out_t


def generate(cfg: dict, seed: int, total_neurons: int | None = None) -> Connectome:
    rng = np.random.default_rng(seed)
    ens = cfg["ensemble"]
    tn = ens["total_neurons"]
    total = total_neurons or tn["default"]
    if tn.get("sample") and not total_neurons:
        total = int(_lognormal_in_range(rng, tn["default"], *tn["range"]))
    sizes = sample_region_sizes(cfg, rng, total)
    df, region_names, tx_names = build_neurons(cfg, rng, sizes)
    rid = df["region_id"].to_numpy()
    starts = {r: int(np.flatnonzero(rid == i)[0]) for i, r in enumerate(region_names)}
    df["gamma_out"] = _gamma(rng, ens["source_degree_shape"], len(df))
    df["gamma_in"] = _gamma(rng, ens["target_degree_shape"], len(df))
    sy = ens["synapses_per_connection"]
    m, r_ = max(sy["mean"] - 1.0, 1e-6), sy["dispersion"]
    sign_all = df["sign"].to_numpy()
    parts, proj_names = [], []
    for pi, proj in enumerate(cfg["projections"]["projections"]):
        k = proj["k"]["default"]
        if ens.get("sample_projection_k", True):
            lo, hi = proj["k"]["range"]
            k = _lognormal_in_range(rng, max(k, 1e-3), max(lo, 1e-3), max(hi, 1e-2))
        s_ids = np.flatnonzero(rid == region_names.index(proj["src"]))
        d_ids = np.flatnonzero(rid == region_names.index(proj["dst"]))
        ds, dd = df.iloc[s_ids], df.iloc[d_ids]
        ss, tt = _wire(rng, proj, k, s_ids, d_ids, ds, dd, ds["gamma_out"].to_numpy(), dd["gamma_in"].to_numpy())
        if len(ss) == 0:
            proj_names.append(proj["name"])
            continue
        src_g, dst_g = s_ids[ss], d_ids[tt]
        keep = src_g != dst_g
        pair = np.unique(np.stack([src_g[keep], dst_g[keep]], 1), axis=0)
        cnt = 1 + rng.negative_binomial(r_, r_ / (r_ + m), len(pair))
        parts.append(pd.DataFrame({
            "src": pair[:, 0].astype(np.int32), "dst": pair[:, 1].astype(np.int32),
            "syn": (cnt * sign_all[pair[:, 0]]).astype(np.int16), "proj": np.int16(pi),
        }))
        proj_names.append(proj["name"])
    edges = pd.concat(parts, ignore_index=True)
    # same pair from two projections: merge by summing so the CSR has no duplicates
    edges = edges.groupby(["src", "dst"], as_index=False).agg(syn=("syn", "sum"), proj=("proj", "first"))
    prov = make_provenance(cfg, seed, total, {"region_sizes": sizes})
    df["region"] = pd.Categorical.from_codes(df["region_id"].to_numpy(), region_names)
    return Connectome(seed, df, edges, region_names, proj_names, prov)
