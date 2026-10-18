"""PPO settings for jumper.ref_free_jump_dir: ref_free_jump's, with its own name and a shorter run (it is warm-started)."""

from __future__ import annotations

from ..ref_free_jump.rl_cfg import NUM_STEPS_PER_ENV, agent_cfg as _base  # noqa: F401


def agent_cfg():
    cfg = _base()
    cfg.experiment_name = "jumper.ref_free_jump_dir"
    cfg.max_iterations = 3000
    return cfg
