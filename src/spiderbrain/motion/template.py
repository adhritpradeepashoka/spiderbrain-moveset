"""Spider jump template: swing (legs IV push) then fling (legs III), as joint-angle-ish time series.

HYPOTHESIS, not data: shapes follow the qualitative description of salticid jumps (rear legs
extend explosively; hydraulic pressure assists). Use motion/dryad.py to replace these with
measured Maratus splendens kinematics (doi:10.5061/dryad.hdr7sqvsp) once downloaded.
Jumper channel names: LR/RR rear legs, LM/RM middle legs, LF/RF claw arms.
"""
from __future__ import annotations

import numpy as np


def jump_template(duration_s=0.5, fs=100, swing_frac=0.35):
    t = np.arange(0, duration_s, 1 / fs)
    u = t / duration_s
    swing = np.clip(u / swing_frac, 0, 1)
    fling = np.clip((u - swing_frac) / (1 - swing_frac), 0, 1)
    ease = lambda x: 3 * x**2 - 2 * x**3
    rear = ease(swing) - 0.3 * ease(fling)        # legs IV: crouch/extend, then release
    mid = 0.2 * ease(swing) + ease(fling)          # legs III: flings late
    arms = 0.5 * ease(fling)
    return {"t": t, "LR": rear, "RR": rear, "LM": mid, "RM": mid, "LF": arms, "RF": arms}


def retarget(template: dict) -> dict:
    """Normalised 0..1 curves -> per-channel arrays; scaling to Jumper joint ranges is robot-specific (TODO)."""
    return {k: np.asarray(v) for k, v in template.items()}
