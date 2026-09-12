"""Shared data-loading and preprocessing utilities for the OASIS project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "oasis_longitudinal.csv"

FEATURES = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
REQUIRED_COLUMNS = ["Subject ID", "Group", *FEATURES]


def load_classification_data():
    """Load data for current-status dementia classification.

    The OASIS dataset contains repeated visits per subject. `Converted` subjects
    are excluded from this task because their group label describes a transition
    over time rather than the clinical status of every visit.
    """
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data = data[data["Group"].isin(["Nondemented", "Demented"])].copy()
    data["Target"] = data["Group"].map({"Nondemented": 0, "Demented": 1}).astype(int)

    X = data[FEATURES]
    y = data["Target"]
    groups = data["Subject ID"]
    return data, X, y, groups
