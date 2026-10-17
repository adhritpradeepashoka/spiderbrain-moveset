from __future__ import annotations

import datetime as dt
import platform

import numpy as np
import scipy

from . import __version__


def make_provenance(cfg: dict, seed: int, total_neurons: int, extra: dict | None = None) -> dict:
    return {
        "generator": "spiderbrain",
        "version": __version__,
        "disclaimer": "SYNTHETIC PRIOR ENSEMBLE MEMBER. Not a measured connectome; no salticid connectome exists.",
        "seed": seed,
        "target_total_neurons": total_neurons,
        "config_hashes": cfg["_hashes"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        **(extra or {}),
    }
