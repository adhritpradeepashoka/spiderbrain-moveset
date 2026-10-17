"""One control tick: camera frame (+depth) -> eyes -> rate net -> readouts -> BrainPad -> PadState.

Same class is used by the mock arena (tests) and by the Jumper launcher, so what is tested here is what runs there.
"""
from __future__ import annotations

import numpy as np

from ..sensors.eyes import EyeFrontEnd
from ..sim.rate import RateNet
from ..sim.readout import Readouts
from .bridge import BrainPad


class BrainController:
    def __init__(self, conn, cfg, control_hz=50.0, substeps=None, backend="auto", eye_gain=5.0, warmup_s=1.0, gaze_follow=True, readout=None):
        self.cfg, self.conn = cfg, conn
        self.dt_ctrl = 1.0 / control_hz
        self.net = RateNet(conn, backend=backend, sensory_cfg=cfg["regions"]["regions"])
        self.substeps = substeps or max(int(round(self.dt_ctrl / self.net.dt)), 1)
        self.eye = EyeFrontEnd(conn, cfg["eyes"], gain=eye_gain)
        self.readouts = Readouts(conn, cfg["movesets"]["readouts"])
        self.pad = BrainPad(cfg["movesets"])
        self.gaze_follow = gaze_follow
        self.external_readout = readout
        self._calibrate(warmup_s)

    def _calibrate(self, secs):
        """Run on a blank (dark) scene to get readout baselines."""
        blank = np.zeros((120, 160, 3), np.float32)
        hist = []
        for i in range(int(secs * 2 / self.dt_ctrl)):
            self._advance(blank, None)
            if i > int(secs / self.dt_ctrl):
                hist.append(self.net.r.copy())
        self.readouts.calibrate(hist)
        self.pad.cmd = {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}

    def _advance(self, frame, depth):
        ext = self.eye(frame, depth)
        for _ in range(self.substeps):
            self.net.step(ext)

    def tick(self, frame, depth_m=None, fovea_contrast=None):
        self._advance(frame, depth_m)
        cmd = (self.external_readout or self.readouts)(self.net.r)
        if self.gaze_follow:
            self.eye.set_gaze(40.0 * cmd.get("scan_az", 0.0))
        self.pad.update(cmd, self.dt_ctrl, distance_m=depth_m, fovea_contrast=fovea_contrast)
        return self.pad.state(), cmd


def render_prey_frame(bearing_rad, distance_m, size_m=0.03, W=160, H=120, fov_deg=120.0):
    """Synthetic camera frame: a bright blob at the prey's bearing (+ = left of robot), size ~ 1/distance."""
    f = np.zeros((H, W, 3), np.float32)
    half = np.deg2rad(fov_deg) / 2
    if abs(bearing_rad) >= half or distance_m <= 0.02:
        return f
    cx = int((0.5 - bearing_rad / (2 * half)) * W)
    r = int(np.clip(size_m / distance_m * W / np.deg2rad(fov_deg) * 1.0, 1, 20))
    cy = H // 2
    f[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = 1.0
    return f
