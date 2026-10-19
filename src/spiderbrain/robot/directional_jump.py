"""Forward-biased jump for Jumper: the stock jump's recorded joint commands plus a leg bias.

Jumper's own jump is a trained policy that tracks a *symmetric* recording (`high_jump_flat`), so it goes straight up and
lands where it started. Replaying the recording's commanded joint angles open-loop (PD, kp=20, no policy) reproduces that
jump in plain MuJoCo (0.157 m rise vs 0.141 m recorded). Adding a smooth bump to a few rear-leg and middle-hip joints
around the push-off turns the same motion into a forward leap. The bias values below were found by a small
evolution-strategy search on a CPU MuJoCo model (see docs/directional-jump.md); they are a starting point, not a result
on Jumper's own simulator until `scripts/directional_jump_check.py` has been run there.

The right side is the mirror of the left (every right joint = -left, except the front-arm shoulder pitch, which is
same-sign), matching the recording's own symmetry.
"""
from __future__ import annotations

import numpy as np

LEFT = ["LR_J0_joint", "LR_J1_joint", "LR_J2_joint", "LM_J1_joint", "LM_J2_joint", "LF_J1_joint"]
RIGHT = ["RR_J0_joint", "RR_J1_joint", "RR_J2_joint", "RM_J1_joint", "RM_J2_joint", "RF_J1_joint"]
MIRROR = [-1, -1, -1, -1, -1, +1]

#: Found on CPU MuJoCo, kp=20, torque clip 1.7464 N*m: ~0.25 m forward, 0.08 m rise, lands upright, no sideways drift.
DEFAULT_PARAMS = {
    "amp": [-0.4957, -0.1507, 0.0392, -0.7601, 0.2074, -0.8947],   # rad, per LEFT joint above
    "centre_s": 0.7223,                                              # bump centre in the recording (push-off is ~0.7 s)
    "width_s": 0.0804,                                               # bump sigma
}
KP, KD = 20.0, 0.5          # the jump task's own gains (the walking gains are kp=10)
RATE_HZ = 200.0             # the recording's command rate; the jump mode runs at 200 Hz


def biased_commands(q_cmd: np.ndarray, joint_names: list[str], params: dict | None = None, scale: float = 1.0) -> np.ndarray:
    """(T, n_joints) commands: the recording's, plus `scale` x the bias. `scale=0` returns the stock jump.

    `params` is either {"amp": [...6], "centre_s", "width_s"} (the original layout over LEFT/RIGHT) or
    {"pairs": [[left, right, mirror_sign, amp], ...], "centre_s", "width_s"} (configs/directional_jump_v3.json).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    pairs = p.get("pairs") or [[l, r, m, a] for l, r, m, a in zip(LEFT, RIGHT, MIRROR, p["amp"])]
    t = np.arange(len(q_cmd)) / RATE_HZ
    bump = np.exp(-0.5 * ((t - p["centre_s"]) / p["width_s"]) ** 2)
    out = np.array(q_cmd, dtype=float, copy=True)
    for l, r, m, a in pairs:
        out[:, joint_names.index(l)] += scale * a * bump
        out[:, joint_names.index(r)] += scale * m * a * bump
    return out


def within_limits(commands: np.ndarray, joint_names: list[str], limits: dict[str, tuple[float, float]]) -> list[str]:
    """Names of joints whose commands leave their range (empty = fine)."""
    bad = []
    for j, name in enumerate(joint_names):
        lo, hi = limits[name]
        if commands[:, j].min() < lo - 1e-6 or commands[:, j].max() > hi + 1e-6:
            bad.append(name)
    return bad


class DirectionalJump:
    """Feeds the biased commands to the player as joint targets, blending in from wherever the legs are."""

    def __init__(self, q_cmd: np.ndarray, joint_names: list[str], params: dict | None = None, scale: float = 1.0, blend_s: float = 0.3):
        self.names = list(joint_names)
        self.cmd = biased_commands(q_cmd, self.names, params, scale)
        self.blend_s = blend_s
        self.duration = len(self.cmd) / RATE_HZ + blend_s
        self._from: np.ndarray | None = None
        self._t0 = 0.0

    def start(self, now_s: float, current_q: np.ndarray) -> None:
        self._t0, self._from = now_s, np.asarray(current_q, dtype=float)

    @property
    def active(self) -> bool:
        return self._from is not None

    def target(self, now_s: float) -> np.ndarray | None:
        """Joint targets (in `joint_names` order) for this instant, or None once the jump is over."""
        if self._from is None:
            return None
        t = now_s - self._t0
        if t >= self.duration:
            self._from = None
            return None
        if t < self.blend_s:
            a = t / self.blend_s
            a = a * a * (3 - 2 * a)
            return (1 - a) * self._from + a * self.cmd[0]
        i = min(int((t - self.blend_s) * RATE_HZ), len(self.cmd) - 1)
        return self.cmd[i]
