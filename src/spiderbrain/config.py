"""Load the YAML priors that define the synthetic brain."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

CONFIG_NAMES = ("regions", "projections", "transmitters", "ensemble", "eyes", "movesets")


def default_config_dir() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parents[2] / "configs", Path.cwd() / "configs"):
        if p.is_dir():
            return p
    raise FileNotFoundError("configs/ directory not found; pass config_dir explicitly")


def load_config(config_dir: str | Path | None = None) -> dict:
    d = Path(config_dir) if config_dir else default_config_dir()
    cfg, hashes = {}, {}
    for name in CONFIG_NAMES:
        raw = (d / f"{name}.yaml").read_bytes()
        cfg[name] = yaml.safe_load(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()[:12]
    cfg["_hashes"] = hashes
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    regions = cfg["regions"]["regions"]
    for p in cfg["projections"]["projections"]:
        for end in ("src", "dst"):
            if p[end] not in regions:
                raise ValueError(f"projection {p['name']}: unknown region {p[end]!r}")
        lo, hi = p["k"]["range"]
        if not lo <= p["k"]["default"] <= hi:
            raise ValueError(f"projection {p['name']}: default k outside range")
    for r, spec in regions.items():
        lo, hi = spec["n"]["range"]
        if not lo <= spec["n"]["default"] <= hi:
            raise ValueError(f"region {r}: default n outside range")
    for r in cfg["transmitters"].get("by_region", {}):
        if r not in regions:
            raise ValueError(f"transmitters.by_region: unknown region {r!r}")


def with_profile(cfg: dict, profile: str | None, total_neurons: int | None = None, n_seeds: int | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    ens = cfg["ensemble"]
    if profile:
        prof = ens["profiles"][profile]
        ens["total_neurons"]["default"] = prof["total_neurons"]
        ens["n_seeds"] = prof["n_seeds"]
    if total_neurons:
        ens["total_neurons"]["default"] = int(total_neurons)
    if n_seeds:
        ens["n_seeds"] = int(n_seeds)
    return cfg
