"""Scripted worlds for the survival crab in Jumper's simulator.

Only one extra object exists in the simulator (the football scene's ball), so the ball plays whichever role the scenario
needs: prey that wanders, a threat that charges, a stationary mate, or nothing. Everything here is geometry and a timer; the
crab's behaviour comes from `SurvivalAgent`. Coordinates are in the robot's INITIAL frame (+x ahead, +y left), metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .agent import Personality, SurvivalAgent

SCENARIOS = ("hunt", "threat", "tun", "court", "ice", "ledge")
BALL_RADIUS = 0.11            # football scene ball, used for the looming (angular size) computation
PARK = (-3.0, -3.0)           # where the ball waits when it has no role


@dataclass
class WorldOut:
    ball: tuple[float, float]
    role: str                  # "prey" | "threat" | "mate" | "none"
    stress: float = 0.0
    mate_receptive: bool = False


def make_agent(name: str, seed: int = 0) -> SurvivalAgent:
    """A crab whose starting mood suits the scenario (so a 20 s clip actually shows the behaviour)."""
    p = Personality.from_seed(seed)
    if name == "court":
        p.flair, p.libido = 0.3, 0.9
    if name == "threat":
        p.boldness = 0.5
    if name == "tun":
        p.patience = 0.4
    if name in ("hunt", "ledge"):
        p.boldness, p.appetite = 0.7, 0.8
    ag = SurvivalAgent(p)
    if name in ("hunt", "ledge"):
        ag.energy = 0.45
    elif name == "court":
        ag.mate_urge, ag.energy = 0.9, 0.9
    else:
        ag.energy = 0.9
    return ag


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class World:
    def __init__(self, name: str, seed: int = 0):
        if name not in SCENARIOS:
            raise ValueError(f"scenario {name!r}; choose from {SCENARIOS}")
        self.name = name
        self.t_threat = 4.0                    # the threat appears this many seconds in
        self._thr = None                       # threat position once spawned
        self._prey_t0 = 0.0
        self._prey_home = (1.1, 0.45)
        self.respawns = 0

    # --- the ball's position in the robot's initial frame, plus the role it plays this tick
    def step(self, s: float, dt: float, robot: tuple[float, float, float], caught: bool = False) -> WorldOut:
        rx, ry, ryaw = robot
        n = self.name
        if n == "hunt":
            if caught:                          # next prey appears a little way off, ahead of where the robot is now
                self.respawns += 1
                ang = ryaw + (0.6 if self.respawns % 2 else -0.6)
                self._prey_home = (rx + 1.2 * math.cos(ang), ry + 1.2 * math.sin(ang))
                self._prey_t0 = s
            u = s - self._prey_t0
            hx, hy = self._prey_home
            return WorldOut((hx + 0.12 * math.sin(0.6 * u), hy + 0.08 * math.sin(1.0 * u)), "prey")
        if n == "threat":
            if s < self.t_threat:
                return WorldOut(PARK, "none")
            if self._thr is None:               # appears 1.6 m away, 40 deg to the left of where the robot faces
                self._thr = (rx + 1.6 * math.cos(ryaw + 0.7), ry + 1.6 * math.sin(ryaw + 0.7))
            tx, ty = self._thr
            d = math.hypot(rx - tx, ry - ty)
            if d > 0.38:                        # charge at 0.9 m/s toward where the robot is now, stop short of it
                k = min(0.9 * dt, d - 0.38) / d
                self._thr = (tx + (rx - tx) * k, ty + (ry - ty) * k)
            return WorldOut(self._thr, "threat")
        if n == "ledge":                        # the prey sits on the ledge, 0.42 m ahead; it does not move
            return WorldOut((0.42, 0.0), "prey")
        if n == "court":
            return WorldOut((1.0, 0.3), "mate", mate_receptive=True)
        if n == "tun":
            if s < 3.0:
                st = 0.15
            elif s < 5.0:
                st = 0.15 + 0.8 * (s - 3.0) / 2.0          # a "dry spell" arrives
            elif s < 13.0:
                st = 0.95
            else:
                st = 0.10                                    # and passes
            return WorldOut(PARK, "none", stress=st)
        return WorldOut(PARK, "none")           # ice: no objects, the floor is the test


def angular_size(dist: float, radius: float = BALL_RADIUS) -> float:
    return 2.0 * math.atan2(radius, max(dist, 1e-3))


def bearing_to(robot: tuple[float, float, float], target: tuple[float, float]) -> tuple[float, float]:
    """(distance, bearing) of `target` from the robot; bearing + = left of the robot's heading."""
    rx, ry, ryaw = robot
    dx, dy = target[0] - rx, target[1] - ry
    return math.hypot(dx, dy), _wrap(math.atan2(dy, dx) - ryaw)
