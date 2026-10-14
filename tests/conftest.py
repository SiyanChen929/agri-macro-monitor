from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest


# Reporting tests must be deterministic and must not require a window server.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def observation_frame() -> Callable[..., pd.DataFrame]:
    """Build a canonical one-series vintage frame from concise row mappings."""

    def factory(*rows: dict[str, Any], series_id: str = "crop_condition") -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for position, supplied in enumerate(rows):
            row = dict(supplied)
            publication = pd.Timestamp(row["publication_date"])
            observation = pd.Timestamp(row["observation_date"])
            records.append(
                {
                    "series_id": row.pop("series_id", series_id),
                    "observation_date": observation,
                    "publication_date": publication,
                    "effective_date": pd.Timestamp(row.pop("effective_date", observation)),
                    "revision_date": pd.Timestamp(row.pop("revision_date", publication)),
                    "value": float(row.pop("value")),
                    "source": row.pop("source", "test_fixture"),
                    "frequency": row.pop("frequency", "weekly"),
                    "unit": row.pop("unit", "index"),
                    "vintage_id": row.pop("vintage_id", f"v{position}"),
                    **row,
                }
            )
        return pd.DataFrame.from_records(records)

    return factory


@pytest.fixture
def market_frame() -> pd.DataFrame:
    """Economically plausible deterministic history with every MVP source."""

    index = pd.bdate_range("2020-01-02", periods=180, name="date")
    t = np.arange(len(index), dtype=float)
    cycle = np.sin(t / 8.0)
    slow_cycle = np.cos(t / 19.0)
    return pd.DataFrame(
        {
            "wheat": 5.0 * np.exp(0.0015 * t + 0.025 * cycle),
            "corn": 4.0 * np.exp(0.0010 * t + 0.018 * slow_cycle),
            "soybeans": 9.0 * np.exp(0.0008 * t + 0.014 * cycle),
            "natural_gas": 2.8 * np.exp(0.0004 * t + 0.020 * slow_cycle),
            "urea": 220.0 * np.exp(0.0007 * t + 0.016 * cycle),
            "cf": 42.0 * np.exp(0.0011 * t + 0.012 * cycle),
            "ntr": 38.0 * np.exp(0.0009 * t + 0.011 * slow_cycle),
            "mos": 31.0 * np.exp(0.0008 * t + 0.013 * cycle),
            "dxy": 95.0 + 0.025 * t + 0.7 * slow_cycle,
            "us_2y": 1.1 + 0.006 * t + 0.08 * cycle,
            "breakeven_5y": 1.7 + 0.002 * t + 0.06 * slow_cycle,
            "crop_condition": 75.0 - 0.035 * t + 1.8 * cycle,
            "yield_revision": 0.8 - 0.012 * t + 0.35 * slow_cycle,
            "exporter_stocks_to_use": 29.0 - 0.025 * t + 0.6 * cycle,
            "policy_stress": 8.0 + 0.055 * t + 1.2 * slow_cycle,
            "weather_stress": 15.0 + 0.075 * t + 2.0 * cycle,
        },
        index=index,
    )


@pytest.fixture
def standardized_factors() -> pd.DataFrame:
    from agri_macro_quant.factors import MVP_FACTOR_COLUMNS

    index = pd.bdate_range("2022-01-03", periods=40, name="date")
    t = np.linspace(-1.5, 1.5, len(index))
    values = {
        name: np.sin(t * (position + 1) / 3.0) + 0.1 * position
        for position, name in enumerate(MVP_FACTOR_COLUMNS)
    }
    return pd.DataFrame(values, index=index)
