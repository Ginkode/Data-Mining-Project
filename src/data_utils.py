"""Shared data-loading and preprocessing utilities for the OASIS project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "oasis_longitudinal.csv"

FEATURES = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
REQUIRED_COLUMNS = ["Subject ID", "Group", *FEATURES]


def prepare_classification_data(data: pd.DataFrame):
    """Prepare a dataframe for current-status dementia classification."""
    clean = data.dropna(subset=REQUIRED_COLUMNS).copy()
    clean = clean[clean["Group"].isin(["Nondemented", "Demented"])].copy()
    clean["Target"] = clean["Group"].map({"Nondemented": 0, "Demented": 1}).astype(int)

    X = clean[FEATURES]
    y = clean["Target"]
    groups = clean["Subject ID"]
    return clean, X, y, groups


def load_classification_data():
    """Load OASIS data from ``data/oasis_longitudinal.csv`` and preprocess it.

    The dataset is intentionally not redistributed with this repository. See
    ``data/README.md`` for setup instructions.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "See data/README.md for download and setup instructions."
        )

    data = pd.read_csv(DATA_PATH)
    return prepare_classification_data(data)
