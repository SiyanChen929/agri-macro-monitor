"""Catalyst calendar utilities.

Dates are configuration inputs rather than silently guessed.  The helper filters,
normalizes, and ranks a supplied calendar, which can come from an official release
adapter in live mode or deterministic fixtures in demo mode.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable, Mapping

import pandas as pd


REQUIRED_COLUMNS = ("date", "event", "affected_regime", "importance")


def upcoming_catalysts(
    as_of: Any,
    events: Iterable[Mapping[str, Any]] | pd.DataFrame | None,
    *,
    horizon_days: int = 45,
) -> pd.DataFrame:
    """Return known catalysts in ``[as_of, as_of + horizon_days]``.

    Unknown dates should not be fabricated.  Callers can surface an empty calendar
    as a data-quality warning and populate it from USDA, NOAA, BLS, BEA, and Federal
    Reserve calendars in a live deployment.
    """

    start = pd.Timestamp(as_of).normalize()
    end = start + timedelta(days=int(horizon_days))
    if events is None:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame = events.copy() if isinstance(events, pd.DataFrame) else pd.DataFrame(list(events))
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Catalyst calendar missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date", "event"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    frame["_importance_order"] = frame["importance"].astype(str).str.lower().map(importance_order).fillna(4)
    return (
        frame.sort_values(["date", "_importance_order", "event"])
        .drop(columns="_importance_order")
        .reset_index(drop=True)
    )

