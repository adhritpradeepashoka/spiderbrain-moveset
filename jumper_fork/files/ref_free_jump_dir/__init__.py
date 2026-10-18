"""jumper.ref_free_jump_dir -- jumper.ref_free_jump with a commanded landing offset.

A fork (see jumper_fork/README.md in the spiderbrain-moveset repo; Apache-2.0 like Jumper).
The actor additionally observes (dx, dy), where it should land relative to where it takes off, in the take-off
body frame (+x forward, +y left), in units of 0.1 m. Nothing else about the task changes.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.ref_free_jump_dir",
    assets=JUMPER_ASSETS,
    description="jumper hexapod jump with a commanded landing offset (fork of ref_free_jump)",
    tags=("locomotion", "jumper", "jump", "directional"),
)
