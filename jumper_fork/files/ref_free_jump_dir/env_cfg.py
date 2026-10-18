"""jumper.ref_free_jump_dir -- ref_free_jump's environment, plus a commanded landing offset.

Built by calling ref_free_jump's own `env_cfg` and editing the result, so every physics and reward decision there
(200 Hz, kp=20, contact penalties, termination) is unchanged. Edits: the go command becomes `DirJumpCommand`; the
actor and critic gain `jump_target` as their LAST term (the warm-start script relies on that); `horizontal_drift`
(which punishes any travel) is replaced by `landing_target` + `flight_target`.

Environment variables (so scripts/ need no change):
  JUMP_TARGET_MAX   largest forward target in metres after the curriculum      (default 0.30)
  JUMP_RAMP_ITERS   "100,1500": curriculum start,end in iterations of THIS run (default 100,1500)
  JUMP_TARGET_X/Y   replay only: fixed landing offset in metres               (default 0.20, 0.0)
"""

from __future__ import annotations

import dataclasses
import os

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..ref_free_jump import env_cfg as base
from ..ref_free_jump.rl_cfg import NUM_STEPS_PER_ENV
from . import mdp as dmdp

LANDING_WEIGHT = 6.0
LANDING_SIGMA_M = 0.10
FLIGHT_WEIGHT = 2.0
FLIGHT_SIGMA_M = 0.25
TARGET_OBS_SCALE_M = 0.10


def env_cfg(asset=None, play: bool = False):
    cfg = base.env_cfg(asset=asset, play=play)

    # -- command ---------------------------------------------------------------------------------------
    old = cfg.commands["jump"]
    fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    r0, r1 = (int(float(x)) for x in os.environ.get("JUMP_RAMP_ITERS", "100,1500").split(","))
    extra = dict(
        target_max_m=float(os.environ.get("JUMP_TARGET_MAX", "0.30")),
        ramp_start_step=r0 * NUM_STEPS_PER_ENV,
        ramp_end_step=r1 * NUM_STEPS_PER_ENV,
    )
    if play:
        extra["fixed_target"] = (
            float(os.environ.get("JUMP_TARGET_X", "0.20")),
            float(os.environ.get("JUMP_TARGET_Y", "0.0")),
        )
    cfg.commands["jump"] = dmdp.DirJumpCommandCfg(**fields, **extra)

    # -- observation: LAST term of both groups ------------------------------------------------------------
    for group in ("actor", "critic"):
        cfg.observations[group].terms["jump_target"] = ObservationTermCfg(
            func=dmdp.jump_target,
            params={"command_name": "jump", "scale": TARGET_OBS_SCALE_M, "mirror_kind": "vec3"},
        )

    # -- rewards ---------------------------------------------------------------------------------------
    cfg.rewards.pop("horizontal_drift")
    cfg.rewards["landing_target"] = RewardTermCfg(
        func=dmdp.landing_target, weight=LANDING_WEIGHT,
        params={"command_name": "jump", "sigma": LANDING_SIGMA_M},
    )
    cfg.rewards["flight_target"] = RewardTermCfg(
        func=dmdp.flight_target, weight=FLIGHT_WEIGHT,
        params={"command_name": "jump", "sigma": FLIGHT_SIGMA_M},
    )
    return cfg


__all__ = ["env_cfg"]
