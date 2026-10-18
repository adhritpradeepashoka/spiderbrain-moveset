"""Forward-jump search WITH time warp (CPU MuJoCo, approx servos at the correct 200 Hz command rate).
x = [3 segment speed factors (crouch / push / rest)] + mirrored edit knots (11 groups x K). Speed >1 = faster than the stock motion.
Everything the search outputs is a q_cmd table at 200 Hz that can be dropped into Jumper's own jump task for verification."""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import cma, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from jumper_phys import JumperPhys
from forward_jump_search import score
import sym_jump_search as S

K = S.K
SEG = (128, 138)          # stock frame indices (200 Hz): end of crouch, end of push
_P = None


def init(root, kp):
    global _P
    S.init(root, kp); _P = S._P


def warp(ref, v):
    N = len(ref); a, b = SEG
    segs = [(0, a, v[0]), (a, b, v[1]), (b, N - 1, v[2])]
    src = []
    for s, e, sp in segs:
        n = max(2, int(round((e - s) / sp)))
        src.append(np.linspace(s, e, n, endpoint=False))
    src = np.concatenate(src + [np.array([N - 1.0])])
    idx = np.arange(N)
    return np.stack([np.interp(src, idx, ref[:, j]) for j in range(22)], 1)


def build(x, P=None):
    P = P or _P
    v = np.clip(x[:3], 0.6, 2.2)
    ref = warp(P.ref["q_cmd"], v)
    N = len(ref)
    knots = np.linspace(0, N - 1, K + 2)
    X = np.asarray(x[3:]).reshape(len(S._G), K)
    delta = np.zeros((N, 22))
    for gi, (l, r, s) in enumerate(S._G):
        D = np.zeros(K + 2); D[1:-1] = X[gi]
        c = np.interp(np.arange(N), knots, D)
        delta[:, l] = c; delta[:, r] = s * c
    return np.clip(ref + delta, P.lo, P.hi)


def fitness(x):
    return -score(_P.rollout(build(x), hold_s=0.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True); ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--gens", type=int, default=200); ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="warp_best"); a = ap.parse_args()
    init(a.jumper, a.kp)
    x0 = np.r_[np.ones(3), np.zeros(len(S._G) * K)]
    sig = np.r_[np.full(3, 0.5), np.ones(len(S._G) * K)] * a.sigma
    es = cma.CMAEvolutionStrategy(x0, 1.0, {"popsize": a.pop, "seed": a.seed, "CMA_stds": list(sig), "verbose": -9})
    t0 = time.time()
    with mp.Pool(2, initializer=init, initargs=(a.jumper, a.kp)) as pool:
        for g in range(a.gens):
            X = es.ask(); es.tell(X, pool.map(fitness, X))
            if g % 5 == 0 or g == a.gens - 1:
                print(f"gen {g:3d} best {-es.best.f:+.3f} speeds {np.round(np.clip(es.best.x[:3],0.6,2.2),2)} ({time.time()-t0:.0f}s)", flush=True)
                np.save(a.out + ".x.npy", es.best.x)
    m = _P.rollout(build(es.best.x), hold_s=0.5)
    print("BEST", json.dumps({k: (None if v is None else round(float(v), 3)) for k, v in m.items()}))
    np.savez(a.out, x=es.best.x, qcmd=build(es.best.x), metrics=json.dumps({k: (None if v is None else float(v)) for k, v in m.items()}))
