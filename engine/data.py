"""Data loading for the CE dashboard. Single source of truth for file paths."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python make_sample_data.py` for sample data, "
            f"or drop your real CSVs into {DATA_DIR}/."
        )
    return pd.read_csv(path)


def load_deals() -> pd.DataFrame:
    return _read("deals.csv")


def load_tranches() -> pd.DataFrame:
    return _read("tranches.csv")


def load_realized() -> pd.DataFrame:
    df = _read("realized_performance.csv")
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df.sort_values(["deal_name", "period_end"])


def deal_names() -> list[str]:
    return sorted(load_deals()["deal_name"].tolist())
