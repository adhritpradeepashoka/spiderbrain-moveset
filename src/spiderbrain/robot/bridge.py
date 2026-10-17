"""Brain readouts -> Jumper gamepad state.

Jumper's GamepadState (controller/xbox.py): lx, ly, rx, ry in -1..1 (ly/ry POSITIVE IS DOWN),
lt/rt 0..1, hat_x/hat_y ints, buttons frozenset of names (A, B, X, Y, LB, RB, view, menu, home, L3, R3).
UNVERIFIED on a live robot/sim: hat sign convention (assumed up = hat_y -1, like ly) and the
turn/twist split of the right stick. Check docs/CONTROLS.md in the Jumper repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PadState:
    """Duck-typed stand-in for jumper's GamepadState (same attribute names)."""
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    buttons: frozenset = field(default_factory=frozenset)

    def __getattr__(self, item):
        if item in {"A", "B", "X", "Y", "LB", "RB", "view", "menu", "home", "L3", "R3"}:
            return item in self.buttons
        raise AttributeError(item)

    @property
    def any_input(self):
        return bool(self.buttons) or self.hat_x or self.hat_y or max(abs(self.lx), abs(self.ly), abs(self.rx), abs(self.ry)) > 0 or max(self.lt, self.rt) > 0


class BrainPad:
    """Pad-like object: `.state()` and `.connected`, as AppPlayer(pad=...) expects (see docs/robot-integration.md)."""

    connected = True

    def __init__(self, movesets_cfg: dict, jump_cooldown_s=3.0):
        self.cfg = movesets_cfg
        self.cmd = {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}
        self.gate_ok = True
        self._last_jump = -1e9
        self._t = 0.0
        self.cooldown = jump_cooldown_s
        self._jump_pulse = 0

    def update(self, cmd: dict, dt: float, distance_m=None, fovea_contrast=None):
        self._t += dt
        self.cmd = cmd
        g = self.cfg["pounce_gate"]
        ok = True
        if distance_m is not None:
            ok &= g["distance_m"][0] <= distance_m <= g["distance_m"][1]
        if fovea_contrast is not None:
            ok &= fovea_contrast >= g["require_fovea_contrast"]
        self.gate_ok = bool(ok)
        if cmd.get("pounce") and ok and self._t - self._last_jump > self.cooldown:
            self._last_jump = self._t
            self._jump_pulse = max(int(0.15 / max(dt, 1e-3)), 1)

    def state(self) -> PadState:
        ax = self.cfg["pad_axes"]
        buttons = set()
        if self._jump_pulse > 0:
            buttons.add("A")
            self._jump_pulse -= 1
        turn = float(max(-1, min(1, self.cmd.get("turn_ccw", 0.0))))
        fwd = float(max(0, min(1, self.cmd.get("forward", 0.0))))
        # the turn lives in the SECOND half of right-stick travel (0.5..1.0) for a given direction
        rx = 0.0
        if abs(turn) > 0.05:
            rx = ax["turn_ccw"]["sign"] * (0.5 + 0.5 * abs(turn)) * (1 if turn > 0 else -1)
        return PadState(ly=ax["forward"]["sign"] * fwd, rx=rx, buttons=frozenset(buttons))
