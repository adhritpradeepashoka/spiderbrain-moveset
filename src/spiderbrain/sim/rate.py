"""Leaky rate network on a generated connectome. numpy/scipy by default; torch (GPU) if available."""
from __future__ import annotations

import numpy as np

from ..generate import Connectome


def _backend(name):
    if name in ("auto", "torch"):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            return "torch", dev
        except Exception:
            if name == "torch":
                raise
    return "numpy", "cpu"


class RateNet:
    """tau dr/dt = -r + relu(gain * W r + ext + bias); W rows = target, scaled so mean |input| ~ O(1)."""

    def __init__(self, c: Connectome, dt=0.005, tau=0.02, gain=None, mod_gain=0.5, backend="auto", seed=0, sensory_cfg=None, rmax=5.0):
        self.rmax = rmax
        self.c, self.dt, self.tau = c, dt, tau
        W = c.weights().astype(np.float32).tocsr()
        # modulatory sources scaled down (they act as slow gain, not drive)
        scale = np.where(c.neurons["modulatory"].to_numpy(), mod_gain, 1.0).astype(np.float32)
        # Histaminergic photoreceptors are sign-inverting in flies (light hyperpolarises them AND the
        # synapse inverts), so net effect: light excites second-order cells. We model that by making
        # sensory sources effectively excitatory here; the stored weights keep the transmitter sign.
        cls = {r: s["cls"] for r, s in sensory_cfg.items()} if sensory_cfg else {}
        sens = np.array([cls.get(self.c.region_names[i]) == "sensory" for i in c.neurons["region_id"].to_numpy()]) if cls else np.zeros(c.n, bool)
        flip = np.where(sens, -1.0, 1.0)
        import scipy.sparse as _sp
        W = W @ _sp.diags((scale * np.where(sens, np.where(c.neurons["sign"].to_numpy() < 0, -1.0, 1.0), 1.0)).astype(np.float32))
        # normalise by typical in-weight so activity neither dies nor explodes
        indeg = np.asarray(abs(W).sum(1)).ravel()
        norm = np.percentile(indeg[indeg > 0], 90) if (indeg > 0).any() else 1.0
        self.gain = (1.0 if gain is None else gain) / norm
        self.W = W.tocsr()
        self.n = c.n
        self.r = np.zeros(self.n, np.float32)
        self.bias = np.zeros(self.n, np.float32)
        self.backend, self.device = _backend(backend)
        if self.backend == "torch":
            import torch
            self._t = torch
            self.Wt = torch.sparse_csr_tensor(
                torch.from_numpy(self.W.indptr.astype(np.int64)), torch.from_numpy(self.W.indices.astype(np.int64)),
                torch.from_numpy(self.W.data), size=self.W.shape).to(self.device)
            self.rt = torch.zeros(self.n, device=self.device)
            self.biast = torch.zeros(self.n, device=self.device)

    def set_bias(self, b: np.ndarray):
        self.bias = b.astype(np.float32)
        if self.backend == "torch":
            self.biast = self._t.from_numpy(self.bias).to(self.device)

    def step(self, ext: np.ndarray) -> np.ndarray:
        a = self.dt / self.tau
        if self.backend == "torch":
            t = self._t
            e = t.from_numpy(ext.astype(np.float32)).to(self.device)
            drive = self.gain * (self.Wt @ self.rt) + e + self.biast
            self.rt = (1 - a) * self.rt + a * t.clamp(drive, 0, self.rmax)
            self.r = self.rt.cpu().numpy()
        else:
            drive = self.gain * (self.W @ self.r) + ext + self.bias
            self.r = (1 - a) * self.r + a * np.clip(drive, 0, self.rmax)
        return self.r

    def reset(self):
        self.r[:] = 0
        if self.backend == "torch":
            self.rt.zero_()

    def run(self, ext_fn, steps):
        for i in range(steps):
            self.step(ext_fn(i))
        return self.r
