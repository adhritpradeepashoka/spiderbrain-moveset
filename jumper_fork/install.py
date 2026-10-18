#!/usr/bin/env python3
"""Install the ref_free_jump_dir task into a Jumper checkout. Idempotent.

    python install.py /path/to/jumper
"""
import shutil, sys
from pathlib import Path

jumper = Path(sys.argv[1]).resolve()
src = Path(__file__).parent / "files" / "ref_free_jump_dir"
dst = jumper / "tasks" / "jumper" / "ref_free_jump_dir"
assert (jumper / "tasks" / "jumper" / "ref_free_jump").is_dir(), f"{jumper} is not a Jumper checkout"
dst.mkdir(exist_ok=True)
for f in src.glob("*.py"):
    shutil.copy2(f, dst / f.name)
init = jumper / "tasks" / "__init__.py"
txt = init.read_text()
line = "from .jumper import ref_free_jump_dir  # noqa: F401,E402\n"
anchor = "from .jumper import ref_free_jump  # noqa: F401,E402\n"
assert anchor in txt, "tasks/__init__.py changed; add the import by hand"
if line not in txt:
    init.write_text(txt.replace(anchor, anchor + line))
print("installed", dst)
