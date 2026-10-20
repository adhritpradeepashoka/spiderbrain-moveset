"""football pitch + a SECOND CRAB (a real Jumper model, passive) to court.

Copied into Jumper's `scenes/` by scripts/kaggle_setup_and_run.sh (Jumper itself is not modified in git; this is a
sideloaded scene).  The mate is a second copy of Jumper's own robot entity.  Nothing drives its motors: play_with_brain.py
poses it each step (facing the suitor, gently swaying), so it is a display partner, not a second brain.
"""
from __future__ import annotations
from dataclasses import replace
from . import football


def scene():
    from mjlab.entity import EntityCfg
    from tasks.jumper.common.constants import HOME, STAND_Z, get_jumper_robot_cfg
    base = football.scene()
    mate = get_jumper_robot_cfg()
    mate.init_state = EntityCfg.InitialStateCfg(pos=(6.0, 4.0, STAND_Z), joint_pos=dict(HOME), joint_vel={".*": 0.0})
    props = dict(base.props)
    props["mate"] = mate
    return replace(base, props=props, nconmax=max(base.nconmax or 0, 400), njmax=max(base.njmax or 0, 1600))
