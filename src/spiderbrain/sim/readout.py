from __future__ import annotations

import numpy as np

from ..generate import Connectome


def pop(c: Connectome, spec: dict) -> np.ndarray:
    return c.region_ids(spec["region"], spec.get("hemi"))


class Readouts:
    """Turn network rates into command scalars using movesets.yaml `readouts`."""

    def __init__(self, c: Connectome, specs: dict):
        self.c, self.specs = c, specs
        self.base: dict[str, float] = {}
        self.hold: dict[str, int] = {}
        self.stats: dict[str, tuple[float, float]] = {}

    def calibrate(self, rates_list):
        """rates_list: iterable of rate vectors from an unstimulated run -> baseline mean/std per readout."""
        R = np.stack(list(rates_list))
        for name, s in self.specs.items():
            v = np.array([self._raw(name, r) for r in R])
            self.base[name] = float(v.mean())
            self.stats[name] = (float(v.mean()), float(v.std() + 1e-6))

    def _raw(self, name, r):
        s = self.specs[name]
        k = s["kind"]
        if k == "contrast":
            return float(r[pop(self.c, s["plus"])].mean() - r[pop(self.c, s["minus"])].mean())
        if k == "mean":
            return float(np.mean([r[pop(self.c, p)].mean() for p in s["pops"]]))
        if k == "zscore_threshold":
            return float(r[pop(self.c, s["pop"])].mean())
        raise ValueError(k)

    def __call__(self, r) -> dict:
        out = {}
        for name, s in self.specs.items():
            raw = self._raw(name, r)
            k = s["kind"]
            if k == "contrast":
                out[name] = float(np.tanh(s.get("scale", 1.0) * raw))
            elif k == "mean":
                v = raw - (self.base.get(name, 0) if s.get("baseline") == "subtract" else 0)
                out[name] = float(np.clip(np.tanh(s.get("scale", 1.0) * v), s.get("min", -1), 1))
            else:
                mu, sd = self.stats.get(name, (0.0, 1.0))
                z = (raw - mu) / sd
                n = self.hold.get(name, 0) + 1 if z > s["threshold"] else 0
                self.hold[name] = n
                out[name] = n >= s.get("hold_ticks", 1)
        return out


class RidgeReadout:
    """Frozen random network + trained linear readout (a reservoir readout).

    The connectome is a random prior, so which neurons 'mean' turn-left is not given by the wiring.
    We therefore fit a ridge regression from a sample of descending/central neuron rates to
    (turn, forward, pounce-ready) on synthetic prey episodes. Only these weights are learned.
    """

    REGIONS = ("dn", "ab", "pn", "mb_out", "al2pl2")

    def __init__(self, c: Connectome, n_feat=800, seed=0, alpha=10.0):
        rng = np.random.default_rng(seed)
        ids = np.concatenate([c.region_ids(r) for r in self.REGIONS])
        self.idx = np.sort(rng.choice(ids, size=min(n_feat, len(ids)), replace=False))
        self.alpha = alpha
        self.W = None

    def features(self, r):
        x = r[self.idx]
        return (x - self.mu) / self.sd if self.W is not None else x

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.W = np.linalg.solve(A, Z.T @ Y)
        return self

    def predict(self, r):
        z = np.r_[(r[self.idx] - self.mu) / self.sd, 1.0]
        return z @ self.W

    def __call__(self, r):
        turn, fwd, _ = self.predict(r)
        # pounce = prey present AND roughly dead ahead (|bearing| < ~15 deg); distance is the non-neural gate
        return {"turn_ccw": float(np.clip(turn, -1, 1)), "forward": float(np.clip(fwd, 0, 1)),
                "pounce": bool(fwd > 0.5 and abs(turn) < 0.33), "scan_az": float(np.clip(turn, -1, 1))}


def fit_ridge_readout(conn, cfg, episodes=60, ticks=120, hz=50.0, seed=0, backend="numpy", verbose=True, p_empty=0.25):
    """Collect network states under random moving prey (and empty scenes), fit the readout.

    Targets: turn = bearing/45 deg (CCW positive); forward = prey present; pounce = prey present and
    within 15 deg of straight ahead. Distance is NOT trained: the pounce distance window is the
    non-neural gate in movesets.yaml. Returns (readout, held-out R^2 per output, split by episode).
    """
    from ..robot.loop import BrainController, render_prey_frame

    rng = np.random.default_rng(seed)
    bc = BrainController(conn, cfg, control_hz=hz, backend=backend, gaze_follow=False)
    ro = RidgeReadout(conn, seed=seed)
    X, Y, E = [], [], []
    for ep in range(episodes):
        bc.net.reset()
        empty = rng.random() < p_empty
        b = rng.uniform(-50, 50)
        d = rng.uniform(0.08, 1.5)
        phase = rng.uniform(0, 6.28)
        for t in range(ticks):
            b = np.clip(b + rng.normal(0, 0.8), -55, 55)
            wig = 6 * np.sin(phase + t / 6)          # prey always moves a little: lateral eyes are motion detectors
            d = np.clip(d + rng.normal(0, 0.01), 0.06, 1.6)
            frame = np.zeros((120, 160, 3), np.float32) if empty else render_prey_frame(np.deg2rad(b + wig), d)
            bc._advance(frame, None if empty else d)
            if t > 25:
                X.append(bc.net.r[ro.idx].copy()); E.append(ep)
                Y.append([0.0, 0.0, 0.0] if empty else [b / 45.0, 1.0, float(abs(b) < 15)])
    X, Y, E = np.array(X), np.array(Y), np.array(E)
    test = np.isin(E, rng.choice(episodes, max(episodes // 5, 1), replace=False))
    ro.fit(X[~test], Y[~test])
    Z = np.c_[(X[test] - ro.mu) / ro.sd, np.ones(test.sum())] @ ro.W
    r2 = 1 - ((Y[test] - Z) ** 2).sum(0) / ((Y[test] - Y[test].mean(0)) ** 2).sum(0)
    ro.fit(X, Y)
    if verbose:
        print("held-out-episode R^2 (turn, forward, pounce):", np.round(r2, 2))
    return ro, r2
