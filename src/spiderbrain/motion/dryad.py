"""Loader stub for the Dryad Maratus splendens kinematics (doi:10.5061/dryad.hdr7sqvsp).

Download is not bundled (licence/size, and the build sandbox can't reach Dryad). Put the CSVs in
data/raw/dryad_maratus/ and call load_csvs(). Column names are NOT assumed: inspect the frame first.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csvs(folder="data/raw/dryad_maratus") -> dict[str, pd.DataFrame]:
    d = Path(folder)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: download the Dryad dataset and extract the CSVs there")
    return {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))}
