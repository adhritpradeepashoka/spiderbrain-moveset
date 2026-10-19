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
    def __init__(self, conn, cfg, control_hz=50.0, brain_hz=50.0, substeps=None, backend="auto", eye_gain=5.0, warmup_s=1.0, gaze_follow=True, readout=None, search_after_s=0.6, search_turn=0.5, drive_pad=True):
        self.cfg, self.conn = cfg, conn
        self.drive_pad = drive_pad          # False: the caller (e.g. the survival agent) updates the pad itself
        self.dt_ctrl = 1.0 / control_hz
        # The brain (and its readout) runs on its own 50 Hz clock whatever the robot's control rate is.
        # Jumper's controller runs ~200 Hz; stepping the eye's frame-difference at 200 Hz would shrink the motion signal 4x.
        self.every = max(int(round(control_hz / brain_hz)), 1)
        self.dt_brain = self.every / control_hz
        self._n = 0
        self.search_after_s, self.search_turn = search_after_s, search_turn
        self._unseen_s, self._search_dir = 0.0, 1.0
        self._cmd = {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}
        self.net = RateNet(conn, backend=backend, sensory_cfg=cfg["regions"]["regions"])
        self.substeps = substeps or max(int(round(self.dt_brain / self.net.dt)), 1)
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
        for i in range(int(secs * 2 / self.dt_brain)):
            self._advance(blank, None)
            if i > int(secs / self.dt_brain):
                hist.append(self.net.r.copy())
        self.readouts.calibrate(hist)
        self.pad.cmd = {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}

    def _advance(self, frame, depth):
        ext = self.eye(frame, depth)
        for _ in range(self.substeps):
            self.net.step(ext)

    def tick(self, frame, depth_m=None, fovea_contrast=None):
        """Call every control tick. The brain recomputes every `self.every` ticks; the pad state is refreshed every tick."""
        if self._n % self.every == 0:
            self._advance(frame, depth_m)
            cmd = dict((self.external_readout or self.readouts)(self.net.r))
            # Search: with nothing in view for a while, turn on the spot toward the side the prey was last seen
            # (default left). A NON-NEURAL shortcut, reported as cmd['search']; real salticids turn to scan too.
            seen = cmd.get("forward", 0.0) > 0.15
            if seen:
                self._unseen_s = 0.0
                if abs(cmd.get("turn_ccw", 0.0)) > 0.05:
                    self._search_dir = 1.0 if cmd["turn_ccw"] > 0 else -1.0
            else:
                self._unseen_s += self.dt_brain
            cmd["search"] = (not seen) and self._unseen_s > self.search_after_s
            if cmd["search"]:
                cmd.update(turn_ccw=self._search_dir * self.search_turn, forward=0.0, pounce=False)
            self._cmd = cmd
            if self.gaze_follow:
                self.eye.set_gaze(40.0 * self._cmd.get("scan_az", 0.0))
        self._n += 1
        if self.drive_pad:
            self.pad.update(self._cmd, self.dt_ctrl, distance_m=depth_m, fovea_contrast=fovea_contrast)
        return self.pad.state(), self._cmd


def render_prey_frame(bearing_rad, distance_m, size_m=0.03, W=160, H=120, fov_deg=130.0):
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
