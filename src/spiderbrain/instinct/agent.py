"""A crab that "knows how to survive": needs, reflexes and a few instincts, arbitrated by priority.

Honest labels (these go in the contest post too):
  * Hunting (stalk -> freeze -> pounce) is the spider-brain pipeline: the caller passes its readout in as `brain_cmd`.
  * The looming escape is a fly-style LGMD-like computation on the threat's angular size (a hand-written model of the
    idea, not the fly connectome).
  * "Tun" shutdown / revival is inspired by tardigrade cryptobiosis; `viability` is a simple stand-in ODE, not the
    user's tardigrade model (plug that in via `viability_rate`).
  * Courtship picks a Jumper gesture/dance; the "mate" is a stand-in object. Only one robot exists in the simulator.
Nothing here is a real animal's brain; it is behaviour logic with biological names.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Personality:
    """Seeded character. Same seed -> same crab."""
    boldness: float = 0.5      # 0 shy .. 1 bold: how close a threat may get before it bolts; how willingly it stalks
    patience: float = 0.5      # freeze time before a pounce; how long it waits before reviving
    libido: float = 0.5        # how fast the mating urge builds
    appetite: float = 0.5      # how fast energy drains
    flair: float = 0.5         # picks the courtship display

    @classmethod
    def from_seed(cls, seed: int) -> "Personality":
        r = random.Random(seed)
        return cls(*(round(r.uniform(0.15, 0.95), 2) for _ in range(5)))


@dataclass
class Percept:
    t: float = 0.0
    prey_dist: float | None = None         # m, None = not seen
    prey_bearing: float = 0.0              # rad, + = left
    threat_dist: float | None = None
    threat_bearing: float = 0.0
    threat_theta: float = 0.0              # angular size of the threat, rad (0 if unseen)
    mate_dist: float | None = None
    mate_bearing: float = 0.0
    mate_receptive: bool = False
    stress: float = 0.0                    # 0..1 environmental stress (e.g. "dry spell"/heat), drives cryptobiosis
    friction: float = 1.0                  # estimated floor friction (1 grippy .. 0.05 ice)
    jump_ready: bool = True
    prey_caught: bool = False              # set by the world for one tick when the pounce lands on the prey


@dataclass
class Intent:
    turn_ccw: float = 0.0
    forward: float = 0.0
    strafe_left: float = 0.0               # + = left, crab scuttle
    pounce: bool = False                   # ask the pad for the stalk-freeze-jump sequence
    crouch: bool = False                   # lower the body (and stay lowered until False)
    gesture: str | None = None             # "hello" | "bow" | "paw" | "salute" | "dance_crab"
    state: str = "wander"
    why: str = ""
    speed_scale: float = 1.0


def looming_signal(theta_hist: list[tuple[float, float]], alpha: float = 2.0) -> float:
    """LGMD-style looming score from (t, angular size) samples: theta_dot * exp(-alpha * theta).

    Rises sharply for an object that is both big-ish and expanding fast (a collision course); stays low for an object
    that is merely near, or merely moving sideways. Uses the last ~0.3 s.
    """
    if len(theta_hist) < 2:
        return 0.0
    t1, th1 = theta_hist[-1]
    t0, th0 = theta_hist[0]
    for t, th in theta_hist:
        if t1 - t <= 0.3:
            t0, th0 = t, th
            break
    if t1 - t0 < 0.05:
        return 0.0
    dth = (th1 - th0) / (t1 - t0)
    return max(dth, 0.0) * math.exp(-alpha * th1)


class SurvivalAgent:
    LOOM_THRESH = 0.12          # score at which the escape reflex fires (shy crabs lower it)
    TUN_STRESS = 0.7            # stress above this long enough -> shutdown
    REVIVE_STRESS = 0.35
    TURN_GAIN, TURN_CAP = 0.3, 0.3        # scale the readout's turn to Jumper's turn rate
    STALK_FAR, STALK_NEAR = 0.5, 0.3      # forward command caps while stalking
    PREY_MEMORY_S = 1.2                   # keep hunting this long after the prey leaves the field of view

    def __init__(self, personality: Personality | None = None, viability_rate=None):
        self.p = personality or Personality()
        self.energy = 0.8
        self.fear = 0.0
        self.mate_urge = 0.2
        self.viability = 1.0
        self.shadow_viability = 1.0          # what viability WOULD be had it never shut down (for the HUD contrast)
        self.metabolism = 1.0
        self.state = "wander"
        self._theta: list[tuple[float, float]] = []
        self._state_t = 0.0
        self._flee_away = 0.0
        self._tun_since = None
        self._revive_t = 0.0
        self._court_stage = 0
        self._last_court = -1e9
        self._viability_rate = viability_rate or (lambda stress, tun: -0.05 * stress * (0.05 if tun else 1.0) + 0.01 * (1 - stress) * (0.0 if tun else 1.0))
        self.events: list[tuple[float, str]] = []
        self.last_loom = 0.0
        self._prey_seen_t, self._last_prey_bearing, self._aligned_t = -1e9, 0.0, 0.0

    # ------------------------------------------------------------------ helpers
    def _enter(self, state: str, t: float, why: str = ""):
        if state != self.state:
            self.events.append((round(t, 2), f"{self.state} -> {state}: {why}"))
            self.state, self._state_t = state, 0.0

    def drives(self) -> dict:
        return {"energy": round(self.energy, 3), "fear": round(self.fear, 3), "mate": round(self.mate_urge, 3),
                "viability": round(self.viability, 3), "shadow": round(self.shadow_viability, 3), "metabolism": round(self.metabolism, 3)}

    # ------------------------------------------------------------------ the tick
    def tick(self, pc: Percept, dt: float, brain_cmd: dict | None = None, pad_phase: str = "stalk") -> Intent:
        self._state_t += dt
        tun = self.state == "tun"
        # --- needs
        self.energy = max(0.0, self.energy - dt * (0.004 + 0.008 * self.p.appetite) * (0.05 if tun else 1.0))
        self.mate_urge = min(1.0, self.mate_urge + dt * 0.012 * (0.3 + self.p.libido) * (0.0 if tun else 1.0))
        self.fear = max(0.0, self.fear - dt * 0.15)
        self.viability = max(0.0, min(1.0, self.viability + dt * self._viability_rate(pc.stress, tun)))
        self.shadow_viability = max(0.0, min(1.0, self.shadow_viability + dt * self._viability_rate(pc.stress, False)))
        self.metabolism = 0.02 if tun else 1.0
        # --- reflex: looming
        if pc.threat_dist is not None and pc.threat_theta > 0:
            self._theta.append((pc.t, pc.threat_theta))
            self._theta = self._theta[-30:]
        else:
            self._theta = []
        loom = looming_signal(self._theta)
        self.last_loom = loom
        thresh = self.LOOM_THRESH * (1.5 - 0.8 * self.p.boldness)
        # a threat that is simply close also frightens, scaled by boldness (not a reflex, a startle)
        close = pc.threat_dist is not None and pc.threat_dist < 0.25 * (1.4 - self.p.boldness)
        if (loom > thresh or close) and not self.state.startswith("flee") and self.state != "tun":
            self.fear = 1.0
            self._flee_away = pc.threat_bearing + math.pi       # face away
            self._enter("flee", pc.t, f"looming {loom:.2f} > {thresh:.2f}" if loom > thresh else "threat too close")
        # --- tardigrade: shutdown / revival
        if self.state != "tun" and pc.stress >= self.TUN_STRESS and self.state != "flee":
            self._enter("tun", pc.t, f"stress {pc.stress:.2f}")
        if self.state == "tun":
            if pc.stress < self.REVIVE_STRESS:
                self._revive_t += dt
                if self._revive_t > 1.0 + 2.0 * self.p.patience:
                    self._revive_t = 0.0
                    self._enter("revive", pc.t, "conditions back")
            else:
                self._revive_t = 0.0
            return Intent(crouch=True, state="tun", why="cryptobiosis: metabolism ~2%", speed_scale=0.0)
        if self.state == "revive":
            if self._state_t > 1.2:
                self._enter("wander", pc.t, "revived")
            return Intent(crouch=False, gesture="hello" if self._state_t < 0.2 else None, state="revive", why="rehydrating")
        # --- flee
        if self.state == "flee":
            if pc.threat_dist is not None:
                self._flee_away = pc.threat_bearing + math.pi
            err = (self._flee_away - 0.0 + math.pi) % (2 * math.pi) - math.pi          # heading error to 'away' in robot frame
            if abs(err) > 0.5 and self._state_t < 2.0:
                return Intent(turn_ccw=max(-1.0, min(1.0, 2.0 * err)), state="flee", why="turn away from looming threat")
            if self._state_t < 2.6 and pc.jump_ready and pc.friction > 0.4:
                self._enter("flee_jump", pc.t, "bolt")
            elif self._state_t >= 2.6 or pc.friction <= 0.4:
                self._enter("flee_run", pc.t, "run (no good jump)" if pc.friction <= 0.4 else "run")
        if self.state == "flee_jump":
            if pad_phase == "stalk" and self._state_t > 0.2:
                self._enter("flee_run", pc.t, "landed")
                self._state_t = 0.0
            return Intent(pounce=True, state="flee_jump", why="escape jump")
        if self.state == "flee_run":
            if self._state_t > 2.5 and (pc.threat_dist is None or pc.threat_dist > 1.0):
                self._enter("wander", pc.t, "safe")
            return Intent(forward=1.0, state="flee_run", why="running", speed_scale=min(1.0, 0.4 + pc.friction))
        # --- floor adaptation modifies everything below
        slick = pc.friction < 0.4
        scale = 0.55 if slick else 1.0
        crouch = slick                          # lower and slow on a slippery floor (the sim showed crouched is faster on ice)
        # --- courtship
        court_ready = (self.mate_urge > 0.6 and self.energy > 0.3 and pc.t - self._last_court > 25.0 and pc.mate_dist is not None)
        if self.state == "court" or court_ready:
            if pc.mate_dist is None:
                self._enter("wander", pc.t, "mate gone")
            else:
                self._enter("court", pc.t, f"mate urge {self.mate_urge:.2f}")
                if pc.mate_dist > 0.55:
                    return Intent(turn_ccw=max(-1, min(1, 2 * pc.mate_bearing)), forward=0.6 if abs(pc.mate_bearing) < 0.5 else 0.0,
                                  state="court", why="approach mate", speed_scale=scale, crouch=crouch)
                names = ["dance_crab", "bow", "hello", "paw", "salute"]
                if self._court_stage == 0:
                    self._court_stage = 1
                    self._stage_t = pc.t
                    g = names[int(self.p.flair * len(names)) % 3] if self.p.flair < 0.6 else "dance_crab"
                    return Intent(gesture=g, state="court", why=f"display: {g}")
                if pc.t - self._stage_t > 5.0:
                    if pc.mate_receptive:
                        self.mate_urge = 0.0
                        self.energy = max(0.0, self.energy - 0.1)
                        self.events.append((round(pc.t, 2), "courtship accepted"))
                    self._court_stage, self._last_court = 0, pc.t
                    self._enter("wander", pc.t, "courtship over")
                return Intent(state="court", why="waiting for response")
        if pc.prey_caught:
            self.energy = min(1.0, self.energy + 0.35)
            self.events.append((round(pc.t, 2), "caught prey: +energy"))
        # --- hunt
        hungry = self.energy < 0.7
        if pc.prey_dist is not None:
            self._prey_seen_t, self._last_prey_bearing = pc.t, pc.prey_bearing
        lost_for = pc.t - self._prey_seen_t
        if (pc.prey_dist is not None or (self.state == "hunt" and lost_for < self.PREY_MEMORY_S)) and hungry and (self.p.boldness > 0.2):
            self._enter("hunt", pc.t, f"energy {self.energy:.2f}, prey in view" if pc.prey_dist is not None else "prey just out of view: turn back")
            cmd = brain_cmd or {}
            # The brain's readout was tuned on a mock arena; Jumper turns far faster per unit of command (about 250 deg/s per 1.0),
            # so its steering is scaled down (TURN_GAIN) and the approach speed is capped so the crab does not overshoot the prey.
            if pc.prey_dist is None:                    # prey slipped out of view: turn back toward where it was last seen
                return Intent(turn_ccw=max(-0.2, min(0.2, 0.6 * self._last_prey_bearing)), state="hunt", why="prey just out of view: turn back", speed_scale=scale, crouch=crouch)
            turn = max(-self.TURN_CAP, min(self.TURN_CAP, self.TURN_GAIN * cmd.get("turn_ccw", 0.0)))
            fwd = cmd.get("forward", 0.0) * scale
            cap = self.STALK_FAR if pc.prey_dist > 0.8 else self.STALK_NEAR
            fwd = min(fwd, cap) if abs(pc.prey_bearing) < 0.35 else 0.0      # reorient first, then advance (spiders do both)
            pounce = bool(cmd.get("pounce"))
            if not pounce and pc.prey_dist < 0.40 and abs(pc.prey_bearing) < 0.2:      # in range and square-on: the documented distance gate
                self._aligned_t += dt
                pounce = self._aligned_t > 0.4
            else:
                self._aligned_t = 0.0 if not pounce else self._aligned_t
            return Intent(turn_ccw=turn, forward=fwd, pounce=pounce and not slick,
                          state="hunt", why="stalk -> freeze -> pounce", speed_scale=scale, crouch=crouch)
        if self.state == "hunt":
            self._enter("wander", pc.t, "prey lost")
        # --- wander
        self._enter("wander", pc.t, "")
        return Intent(turn_ccw=0.12 * math.sin(pc.t * 0.35), forward=0.45 * scale, state="wander", why="explore", speed_scale=scale, crouch=crouch)


class FrictionEstimator:
    """Proprioceptive slip sense: how much of the commanded walking speed actually happens?

    ratio = measured speed / (commanded fraction x nominal speed). Plain floor ~1.0, ice ~0.4 in Jumper's sim (see
    docs/floor-test-results.md). Maps to a 0..1 'grip' number; the agent treats < 0.4 as slippery. It needs a second or
    so of walking before it has an opinion (until then it reports 1.0).
    """

    def __init__(self, nominal_speed=0.75, window_s=1.5):
        self.v0, self.window = nominal_speed, window_s
        self._hist: list[tuple[float, float, float, float]] = []   # t, x, y, commanded forward fraction
        self.grip = 1.0

    def update(self, t: float, x: float, y: float, cmd_forward: float, walking_ok: bool) -> float:
        if not walking_ok or cmd_forward < 0.4:
            self._hist = []
            return self.grip
        self._hist.append((t, x, y, cmd_forward))
        while self._hist and t - self._hist[0][0] > self.window:
            self._hist.pop(0)
        if t - self._hist[0][0] >= self.window * 0.8:
            t0, x0, y0, _ = self._hist[0]
            speed = math.hypot(x - x0, y - y0) / max(t - t0, 1e-6)
            exp = self.v0 * sum(h[3] for h in self._hist) / len(self._hist)
            ratio = speed / max(exp, 1e-6)
            self.grip = max(0.05, min(1.0, (ratio - 0.25) / 0.6))
        return self.grip
