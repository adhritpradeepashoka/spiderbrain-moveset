"""Symmetric low-dim open-loop search for a FORWARD jump (CPU MuJoCo, approximate servos; see jumper_phys.py).
Edits are mirrored left/right (so the jump stays straight): 10 joint groups x K knots, plus a static 'lean' bias on the
take-off phase. Score = forward distance covered in the air, upright landing, take-off within 5 cm of start."""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import cma, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from jumper_phys import JumperPhys
from forward_jump_search import score

K = 6
_P = None; _G = None


def init(root, kp):
    global _P, _G
    _P = JumperPhys(root, kp=kp)
    J = _P.joints; q = _P.ref["q_cmd"]
    groups = []
    for name in J:
        if name.startswith("L"):
            r = "R" + name[1:]
            s = np.sign(np.sum(q[:, J.index(name)] * q[:, J.index(r)])) or -1.0
            groups.append((J.index(name), J.index(r), s))
    _G = groups   # 11 groups (J4 front included)


def build(x, P=None):
    P = P or _P
    ref = P.ref["q_cmd"]; N = len(ref)
    knots = np.linspace(0, N - 1, K + 2)
    X = np.asarray(x).reshape(len(_G), K)
    delta = np.zeros((N, 22))
    for gi, (l, r, s) in enumerate(_G):
        D = np.zeros(K + 2); D[1:-1] = X[gi]
        c = np.interp(np.arange(N), knots, D)
        delta[:, l] = c; delta[:, r] = s * c
    return np.clip(ref + delta, P.lo, P.hi)


def fitness(x):
    return -score(_P.rollout(build(x), hold_s=0.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True); ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--gens", type=int, default=80); ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x0", default=None); ap.add_argument("--out", default="sym_best"); a = ap.parse_args()
    init(a.jumper, a.kp)
    es = cma.CMAEvolutionStrategy(np.load(a.x0) if a.x0 else np.zeros(len(_G) * K), a.sigma, {"popsize": a.pop, "seed": a.seed, "bounds": [-1.2, 1.2], "verbose": -9})
    t0 = time.time()
    with mp.Pool(2, initializer=init, initargs=(a.jumper, a.kp)) as pool:
        for g in range(a.gens):
            X = es.ask(); es.tell(X, pool.map(fitness, X))
            if g % 5 == 0 or g == a.gens - 1:
                print(f"gen {g:3d} best {-es.best.f:+.3f} ({time.time()-t0:.0f}s)", flush=True)
                np.save(a.out + ".x.npy", es.best.x)
    m = _P.rollout(build(es.best.x), hold_s=0.5)
    print("BEST", json.dumps({k: (None if v is None else round(float(v), 3)) for k, v in m.items()}))
    np.savez(a.out, x=es.best.x, qcmd=build(es.best.x), metrics=json.dumps({k: (None if v is None else float(v)) for k, v in m.items()}))
