"""Write a copy of Jumper's high_jump_flat.npz whose q_cmd table is replaced by a searched one.
Usage (Kaggle): python experiments/make_ref_npz.py <search.npz> <jumper_root>
Backs the original up to high_jump_flat.orig.npz (once). Replay with:
  python scripts/play.py --task jumper.jump --agent zero --headless --steps N --video out.mp4
Zero action = the controller sends exactly q_cmd, so this plays the searched motion through Jumper's own servo model."""
import shutil, sys
from pathlib import Path
import numpy as np
res, root = Path(sys.argv[1]), Path(sys.argv[2])
ref = root / "tasks/jumper/jump/ref/high_jump_flat.npz"
orig = ref.with_name("high_jump_flat.orig.npz")
if not orig.exists():
    shutil.copy(ref, orig)
d = dict(np.load(orig, allow_pickle=False))
q = np.load(res, allow_pickle=False)["qcmd"].astype(d["q_cmd"].dtype)
print("q_cmd", d["q_cmd"].shape, "->", q.shape)
d["q_cmd"] = q
np.savez(ref, **d)
print("wrote", ref)
