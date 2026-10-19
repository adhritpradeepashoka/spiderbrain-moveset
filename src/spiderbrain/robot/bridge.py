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

    def __init__(self, movesets_cfg: dict, jump_cooldown_s=8.0, settle_s=0.8, hold_s=2.2, jump_style="stock"):
        self.cfg = movesets_cfg
        self.cmd = {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}
        self.gate_ok = True
        self._last_jump = -1e9
        self._t = 0.0
        self.cooldown = jump_cooldown_s
        self._jump_pulse = 0
        # Stalk -> freeze -> pounce -> hold. Jumper's jump runs by itself once its mode is entered, so the pad must be
        # STILL when A is pressed (a walking robot ignores or fumbles it) and must stay still while the jump plays.
        self.settle_s, self.hold_s = settle_s, hold_s
        self._phase, self._phase_t = "stalk", 0.0
        # "stock": press A, Jumper's own (vertical) jump runs. "directional": raise `jump_request` instead and let the
        # player (scripts/play_with_brain.py) drive the joints with the forward-biased jump.
        self.jump_style = jump_style
        self.jump_request = False
        # Survival-crab extras: body height (R3 + right stick, then release to keep it), gestures (d-pad), dances (Menu + d-pad).
        self._h_state, self._h_move, self._h_t = "stand", None, 0.0
        self._gest, self._gest_t, self._gest_pulse, self._leave_pulse = None, 0.0, 0, 0
        self.gesture_hold = {"hello": 7.5, "bow": 9.0, "paw": 6.7, "salute": 6.3, "dance_crab": 9.0}
        self.lower_s, self.raise_s = 0.45, 0.95

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def crouched(self) -> bool:
        return self._h_state == "low" and self._h_move is None

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
        self._phase_t += dt
        if self._phase == "stalk":
            if cmd.get("pounce") and (ok or cmd.get("escape")) and self._t - self._last_jump > self.cooldown:
                self._phase, self._phase_t = "settle", 0.0          # freeze first
        elif self._phase == "settle":
            if self._phase_t >= self.settle_s:
                self._last_jump = self._t
                if self.jump_style == "directional":
                    self.jump_request = True
                else:
                    self._jump_pulse = max(int(0.15 / max(dt, 1e-3)), 1)
                self._phase, self._phase_t = "hold", 0.0
        elif self._phase == "hold":
            if self._phase_t >= self.hold_s:
                self._phase, self._phase_t = "stalk", 0.0
        elif self._phase == "gesture":
            if self._phase_t >= self.gesture_hold.get(self._gest, 7.0):
                if self._gest == "dance_crab":
                    self._leave_pulse = max(int(0.25 / max(dt, 1e-3)), 1)     # Menu pressed and let go leaves a dance
                self._phase, self._phase_t, self._gest = "stalk", 0.0, None
        # body height: only while standing around (never mid-jump or mid-gesture)
        if self._phase == "stalk":
            want = "low" if cmd.get("crouch") else "stand"
            if self._h_move is None and want != self._h_state:
                self._h_move, self._h_t = ("lower" if want == "low" else "raise"), 0.0
            if self._h_move is not None:
                self._h_t += dt
                if self._h_t >= (self.lower_s if self._h_move == "lower" else self.raise_s):
                    self._h_state, self._h_move = ("low" if self._h_move == "lower" else "stand"), None
            g = cmd.get("gesture")
            if g and self._h_move is None and not self.cmd.get("_gesture_busy"):
                self._gest, self._phase, self._phase_t = g, "gesture", 0.0
                self._gest_pulse = max(int(0.2 / max(dt, 1e-3)), 1)

    def state(self) -> PadState:
        ax = self.cfg["pad_axes"]
        buttons = set()
        if self._jump_pulse > 0:
            buttons.add("A")
            self._jump_pulse -= 1
        hat_x = hat_y = 0
        ry = 0.0
        if self._gest_pulse > 0 and self._gest:
            self._gest_pulse -= 1
            hat_y = {"hello": -1, "bow": 1, "dance_crab": -1}.get(self._gest, 0)
            hat_x = {"paw": -1, "salute": 1}.get(self._gest, 0)
            if self._gest == "dance_crab":
                buttons.add("menu")
        if self._leave_pulse > 0:
            self._leave_pulse -= 1
            buttons.add("menu")
        if self._h_move is not None:
            buttons.add("R3")
            ry = 1.0 if self._h_move == "lower" else -1.0          # ry positive = down
        turn = float(max(-1, min(1, self.cmd.get("turn_ccw", 0.0))))
        fwd = float(max(0, min(1, self.cmd.get("forward", 0.0))))
        strafe = float(max(-1, min(1, self.cmd.get("strafe_left", 0.0))))
        if self._phase != "stalk" or self._h_move is not None:      # frozen while settling, jumping, in a gesture, or changing height
            turn, fwd, strafe = 0.0, 0.0, 0.0
        # the turn lives in the SECOND half of right-stick travel (0.5..1.0) for a given direction
        rx = 0.0
        if abs(turn) > 0.05:
            rx = ax["turn_ccw"]["sign"] * (0.5 + 0.5 * abs(turn)) * (1 if turn > 0 else -1)
        return PadState(ly=ax["forward"]["sign"] * fwd, lx=ax["strafe_left"]["sign"] * strafe, rx=rx, ry=ry,
                        hat_x=hat_x, hat_y=hat_y, buttons=frozenset(buttons))
