"""Kinematic stand-in arena for the survival-instinct logic (no Jumper, no GPU): prey, a charging threat, a dry spell, a mate,
a slippery patch. It tests the BEHAVIOUR LOGIC only; the robot here is a unicycle with a 25 cm jump and timed gestures.

    python scripts/instinct_sim.py --seed 3 --csv instinct_sim.csv
"""
import argparse, csv, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spiderbrain.instinct import Personality, Percept, SurvivalAgent

DT = 0.02


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_scenario(seed=3, T=110.0, log=None):
    ag = SurvivalAgent(Personality.from_seed(seed))
    ag.energy = 0.55
    x = y = th = 0.0
    prey = [1.2, 0.5]
    threat = None
    lock_until, lock_kind = 0.0, None
    pad_phase, phase_t = "stalk", 0.0
    t = 0.0
    rows, theta_prev = [], []
    last_pounce = -1e9
    caught = 0
    prey_gone_until = -1.0
    mate_pos = None
    while t < T:
        # ---- world script
        stress = 0.95 if 52.0 <= t < 66.0 else (0.2 if t >= 66.0 else 0.1)
        friction = 0.08 if 90.0 <= t < 105.0 else 1.0
        if abs(t - 25.0) < DT / 2:                       # a threat charges from 1.6 m ahead-right at 1.6 m/s
            threat = [x + 1.6 * math.cos(th - 0.5), y + 1.6 * math.sin(th - 0.5), x, y]
        if mate_pos is None and t >= 72.0:
            mate_pos = [x + 1.5 * math.cos(th + 0.2), y + 1.5 * math.sin(th + 0.2)]
        mate = mate_pos if 72.0 <= t < 105.0 else None
        # ---- percepts
        def polar(p):
            dx, dy = p[0] - x, p[1] - y
            return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - th)
        pd, pb = polar(prey)
        just_caught = False
        if pd < 0.18 and prey_gone_until < t:
            just_caught = True
            caught += 1; prey_gone_until = t + 6.0
            prey[0], prey[1] = x + 1.4 * math.cos(th + 0.4), y + 1.4 * math.sin(th + 0.4)
        pc = Percept(t=t, stress=stress, friction=friction, jump_ready=(t - last_pounce > 8.0), prey_caught=just_caught)
        if abs(pb) < 1.1 and pd < 2.5 and t >= prey_gone_until:
            pc.prey_dist, pc.prey_bearing = pd, pb
        if threat is not None:
            tx, ty = threat[0], threat[1]
            td, tb = polar([tx, ty])
            if td < 3.0 and abs(tb) < 1.3:
                pc.threat_dist, pc.threat_bearing, pc.threat_theta = td, tb, 2 * math.atan2(0.11, max(td, 0.05))
        if mate is not None:
            md, mb = polar(mate)
            pc.mate_dist, pc.mate_bearing, pc.mate_receptive = md, mb, True
        # stand-in for the spider-brain readout: turn toward prey, go, pounce when in range
        cmd = {"turn_ccw": max(-1, min(1, 2 * pb)), "forward": 1.0 if pd > 0.4 else 0.0, "pounce": 0.05 < pd < 0.4 and abs(pb) < 0.3}
        it = ag.tick(pc, DT, cmd, pad_phase)
        # ---- pad/phase emulation: pounce = 0.8 s freeze, jump (25 cm forward over 0.6 s), 1.2 s hold
        if pad_phase == "stalk":
            if it.pounce and pc.jump_ready:
                pad_phase, phase_t = "settle", 0.0
            if it.gesture and t > lock_until:
                pad_phase, phase_t, lock_kind = "gesture", 0.0, it.gesture
        else:
            phase_t += DT
            if pad_phase == "settle" and phase_t >= 0.8:
                pad_phase, phase_t, last_pounce = "jump", 0.0, t
            elif pad_phase == "jump":
                x += 0.25 / 0.6 * DT * math.cos(th); y += 0.25 / 0.6 * DT * math.sin(th)
                if phase_t >= 0.6:
                    pad_phase, phase_t = "hold", 0.0
            elif pad_phase == "hold" and phase_t >= 1.2:
                pad_phase = "stalk"
            elif pad_phase == "gesture" and phase_t >= 4.0:
                pad_phase = "stalk"
        if pad_phase == "stalk" and not (it.state == "tun"):
            v = 0.75 * it.forward * it.speed_scale
            th += 1.2 * it.turn_ccw * DT
            x += v * math.cos(th) * DT; y += v * math.sin(th) * DT
        if threat is not None:                                         # the threat charges at us
            dx, dy = x - threat[0], y - threat[1]
            d = math.hypot(dx, dy) or 1e-6
            sp = 1.6 if t < 33.0 else 0.0
            threat[0] += sp * dx / d * DT; threat[1] += sp * dy / d * DT
            if t >= 33.0:
                threat[0] -= 0.8 * dx / d * DT; threat[1] -= 0.8 * dy / d * DT
        rows.append((round(t, 2), it.state, it.why, round(x, 3), round(y, 3), round(math.degrees(th), 1), *ag.drives().values(), pad_phase, it.crouch, it.gesture or ""))
        t += DT
    if log:
        with open(log, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "state", "why", "x", "y", "yaw_deg", "energy", "fear", "mate", "viability", "shadow_viability", "metabolism", "pad_phase", "crouch", "gesture"]); w.writerows(rows)
    return ag, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    ag, rows = run_scenario(a.seed, log=a.csv)
    print(f"personality: {ag.p}")
    for t, e in ag.events: print(f"{t:7.2f}s  {e}")
    print("final drives:", ag.drives())
