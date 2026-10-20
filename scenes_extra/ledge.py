"""football pitch + a low LEDGE (a 4 cm raised slab) with the prey ball sitting on it.

Sideloaded into Jumper's `scenes/` by scripts/kaggle_setup_and_run.sh.  The slab is a heavy free body that
play_with_brain.py re-poses every step (like the ball), so it stays put; the crab has to pounce UP onto it.
"""
from __future__ import annotations
from dataclasses import replace
from . import football

STEP_HALF = (0.35, 0.40, 0.02)       # half extents (m): 0.70 long, 0.80 wide, 4 cm tall


def _step_spec():
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="step", pos=[0.0, 0.0, 0.0])
    body.add_freejoint()
    body.add_geom(name="step", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(STEP_HALF), mass=40.0,
                  rgba=[0.86, 0.55, 0.20, 1.0], friction=[1.0, 0.005, 0.0001], contype=1, conaffinity=1, group=2)
    return spec


def scene():
    from mjlab.entity import EntityCfg
    base = football.scene()
    step = EntityCfg(spec_fn=_step_spec, init_state=EntityCfg.InitialStateCfg(pos=(6.0, 4.0, STEP_HALF[2])))
    props = dict(base.props)
    props["step"] = step
    return replace(base, props=props, nconmax=max(base.nconmax or 0, 300), njmax=max(base.njmax or 0, 1200))
