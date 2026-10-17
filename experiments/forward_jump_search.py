"""Can Jumper's servos jump FORWARD? Open-loop CMA-ES over a low-dimensional edit of Jumper's own reference jump.

Usage: python forward_jump_search.py --jumper /path/to/jumper --kp 20 --gens 120 --out best_kp20.npz
Edit = piecewise-linear joint-offset curves (K knots over the reference duration, zero at both ends) added to the
reference q_cmd. Score = forward travel of the base after landing and settling, if the robot stays upright.
This is a feasibility probe of the PHYSICS (my servo approximation), not a controller: open loop, no disturbances.
"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import cma, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from jumper_phys import JumperPhys

K = 8
_P = None


def init(root, kp):
    global _P
    _P = JumperPhys(root, kp=kp)


def build(x, P):
    ref = P.ref["q_cmd"]
    N = len(ref)
    knots = np.linspace(0, N - 1, K + 2)
    D = np.zeros((K + 2, 22))
    D[1:-1] = np.asarray(x).reshape(K, 22)
    delta = np.stack([np.interp(np.arange(N), knots, D[:, j]) for j in range(22)], 1)
    return np.clip(ref + delta, P.lo, P.hi)


def score(m):
    if m["td_fwd"] is None or m["to_fwd"] is None:
        return -1.0
    air = m["td_fwd"] - m["to_fwd"]            # distance covered IN THE AIR only
    if abs(m["to_fwd"]) > 0.05 or m["flight_s"] < 0.15:   # no sliding/walking before takeoff
        return -0.5 + 0.1 * air
    upright = m["end_tilt"] < 0.5 and m["end_z"] > 0.06 and m["min_z"] > 0.03 and m["max_tilt"] < 1.2
    base = air if upright else 0.25 * (m["td_fwd"] or 0.0) - 0.2
    return base - 0.5 * abs(m["end_side"]) - 0.1 * m["max_tilt"] * (0 if upright else 1)


def fitness(x):
    m = _P.rollout(build(x, _P), hold_s=0.5)
    return -score(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True)
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="best.npz")
    a = ap.parse_args()
    init(a.jumper, a.kp)
    es = cma.CMAEvolutionStrategy(np.zeros(K * 22), a.sigma, {"popsize": a.pop, "seed": a.seed, "bounds": [-1.2, 1.2], "verbose": -9})
    t0 = time.time()
    with mp.Pool(2, initializer=init, initargs=(a.jumper, a.kp)) as pool:
        for g in range(a.gens):
            X = es.ask()
            F = pool.map(fitness, X)
            es.tell(X, F)
            if g % 5 == 0 or g == a.gens - 1:
                print(f"gen {g:3d} best {-es.best.f:+.3f} m  ({time.time()-t0:.0f}s)", flush=True)
                np.save(a.out + ".x.npy", es.best.x)
    xb = es.best.x
    m = _P.rollout(build(xb, _P), hold_s=0.5)
    print("BEST", json.dumps({k: (None if v is None else round(float(v), 3)) for k, v in m.items()}))
    np.savez(a.out, x=xb, qcmd=build(xb, _P), metrics=json.dumps({k: (None if v is None else float(v)) for k, v in m.items()}))
