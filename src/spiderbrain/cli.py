from __future__ import annotations

import argparse
import time

from .config import load_config, with_profile
from .generate import generate
from .io import save


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spiderbrain", description="Synthetic jumping-spider connectome prior ensemble")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="sample connectomes")
    g.add_argument("--profile", choices=["tiny", "default", "large"], default="default")
    g.add_argument("--neurons", type=int)
    g.add_argument("--seeds", type=int)
    g.add_argument("--seed0", type=int)
    g.add_argument("--out", default="out")
    g.add_argument("--config-dir")
    a = ap.parse_args(argv)
    cfg = with_profile(load_config(a.config_dir), a.profile, a.neurons, a.seeds)
    ens = cfg["ensemble"]
    s0 = ens["seed0"] if a.seed0 is None else a.seed0
    for seed in range(s0, s0 + ens["n_seeds"]):
        t = time.time()
        c = generate(cfg, seed, ens["total_neurons"]["default"])
        d = save(c, a.out, cfg)
        print(f"seed {seed}: {c.n} neurons, {len(c.edges)} edges, {time.time()-t:.1f}s -> {d}")


if __name__ == "__main__":
    main()
