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
