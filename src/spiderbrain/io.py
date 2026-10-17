from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from .generate import Connectome


def save(c: Connectome, out: str | Path, cfg: dict) -> Path:
    d = Path(out) / f"seed{c.seed:04d}"
    d.mkdir(parents=True, exist_ok=True)
    o = cfg["ensemble"]["outputs"]
    if o.get("neurons_parquet", True):
        c.neurons.drop(columns=["region"]).to_parquet(d / "neurons.parquet")
    if o.get("edges_parquet", True):
        c.edges.to_parquet(d / "edges.parquet")
    if o.get("weights_npz", True):
        sp.save_npz(d / "weights.npz", c.weights())
    if o.get("region_graph_graphml", True):
        _region_graphml(c, d / "region_graph.graphml")
    prov = dict(c.provenance, region_names=c.region_names, projections=c.proj_names,
                n_neurons=c.n, n_edges=len(c.edges))
    (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    return d


def load(path: str | Path) -> Connectome:
    d = Path(path)
    prov = json.loads((d / "provenance.json").read_text())
    neurons = pd.read_parquet(d / "neurons.parquet")
    neurons["region"] = pd.Categorical.from_codes(neurons["region_id"].to_numpy(), prov["region_names"])
    edges = pd.read_parquet(d / "edges.parquet")
    return Connectome(prov["seed"], neurons, edges, prov["region_names"], prov["projections"], prov)


def _region_graphml(c: Connectome, path: Path) -> None:
    rid = c.neurons["region_id"].to_numpy()
    e = c.edges
    agg = (pd.DataFrame({"s": rid[e["src"]], "d": rid[e["dst"]], "w": e["syn"].astype(int)})
           .groupby(["s", "d"]).agg(n=("w", "size"), syn=("w", "sum")).reset_index())
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
         '<key id="n" for="edge" attr.name="connections" attr.type="int"/>',
         '<key id="s" for="edge" attr.name="signed_synapses" attr.type="int"/>',
         '<graph edgedefault="directed">']
    L += [f'<node id="{r}"/>' for r in c.region_names]
    for row in agg.itertuples():
        L.append(f'<edge source="{c.region_names[row.s]}" target="{c.region_names[row.d]}">'
                 f'<data key="n">{row.n}</data><data key="s">{row.syn}</data></edge>')
    L += ["</graph>", "</graphml>"]
    path.write_text("\n".join(L))
