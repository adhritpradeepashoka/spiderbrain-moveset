"""Camera frame -> drive for photoreceptor populations. Pure numpy; frame is HxWx3 uint8/float."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ..generate import Connectome


class EyeFrontEnd:
    def __init__(self, c: Connectome, eyes_cfg: dict, gain=1.0):
        self.c, self.cfg, self.gain = c, eyes_cfg, gain
        self.prev = None
        self.gaze = np.zeros(2)  # azimuth, elevation in degrees
        self.ids = {k: {h: c.region_ids(k, h) for h in (0, 1)} for k in ("ame_pr", "ale_pr", "ple_pr", "pme_pr")}

    def set_gaze(self, az_deg, el_deg=0.0):
        r = self.cfg["ame"]["gaze_range_deg"]
        self.gaze = np.array([np.clip(az_deg, *r["azimuth"]), np.clip(el_deg, *r["elevation"])])

    @staticmethod
    def _gray(f):
        f = f.astype(np.float32)
        if f.max() > 1.5:
            f = f / 255.0
        return f

    def _spread(self, vec, ids):
        """map a 1-D array of drives onto receptors by their `pos` (retinotopic)."""
        pos = self.c.neurons["pos"].to_numpy()[ids]
        return np.interp(pos, np.linspace(0, 1, len(vec)), vec).astype(np.float32)

    def __call__(self, frame, depth_m=None):
        f = self._gray(frame)
        H, W = f.shape[:2]
        g = f.mean(2)
        drive = np.zeros(self.c.n, np.float32)
        ame = self.cfg["ame"]
        # --- principal eye: strip around gaze
        cx = int((0.5 + self.gaze[0] / 120.0) * W)  # 162 deg diag FOV ~ 120 deg horizontal (assumption)
        cy = int((0.5 - self.gaze[1] / 90.0) * H)
        rows, cols = ame["strip_pixels"]
        hh = max(int(H * ame["strip_height_deg"] / 90.0 / 2), 2)
        ww = max(int(W * ame["strip_width_deg"] / 120.0 / 2), 1)
        y0, y1 = max(cy - hh, 0), min(cy + hh, H)
        x0, x1 = max(cx - ww, 0), min(cx + ww, W)
        patch = f[y0:y1, x0:x1]
        if patch.size == 0:
            patch = np.zeros((2, 2, 3), np.float32)
        patch = zoom(patch, (rows / patch.shape[0], cols / patch.shape[1], 1), order=1)[:rows, :cols]
        d = ame["defocus"]
        sig = 0.0
        if depth_m is not None:
            sig = float(np.clip(d["gain_px_m"] * abs(1 / max(depth_m, 0.02) - 1 / d["d_ref_m"]), 0, d["max_sigma_px"]))
        layers = [patch[..., 1], gaussian_filter(patch[..., 1], sig) if sig > 0 else patch[..., 1],
                  patch[..., 2], gaussian_filter(patch.mean(2), 2.0)]
        for h in (0, 1):
            ids = self.c.region_ids("ame_pr", h)
            lay = self.c.neurons["layer"].to_numpy()[ids]
            pos = self.c.neurons["pos"].to_numpy()[ids]
            for li, L in enumerate(layers):
                m = lay == li
                if m.any():
                    colmean = L.mean(1)
                    drive[ids[m]] = np.interp(pos[m], np.linspace(0, 1, len(colmean)), colmean)
        # --- secondary eyes: motion (frame difference) split by image half
        sec = self.cfg["secondary"]
        mot = np.zeros_like(g) if self.prev is None else np.abs(g - self.prev)
        self.prev = g
        bins = sec["motion"]["bins"]
        half = W // 2
        for h, sl in ((0, slice(0, half)), (1, slice(half, W))):  # left image half -> left eye
            prof = mot[:, sl].mean(0)
            prof = np.interp(np.linspace(0, len(prof) - 1, bins), np.arange(len(prof)), prof)
            for k in ("ale_pr", "ple_pr"):
                ids = self.ids[k][h]
                drive[ids] = self._spread(prof, ids) * sec["motion"]["frame_diff_gain"] * 10
            ids = self.ids["pme_pr"][h]  # motion-blind: luminance only
            drive[ids] = g[:, sl].mean()
        return drive * self.gain
