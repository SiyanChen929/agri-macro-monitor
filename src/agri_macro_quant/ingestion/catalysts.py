"""Generate a deterministic catalyst fixture from configuration rules."""

from __future__ import annotations

import calendar as month_calendar
from typing import Any, Mapping

import pandas as pd


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _following_business_day(value: pd.Timestamp) -> pd.Timestamp:
    result = value.normalize()
    while result.weekday() >= 5:
        result += pd.Timedelta(days=1)
    return result


def build_catalyst_calendar(
    catalyst_config: Mapping[str, Any],
    *,
    as_of: Any,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """Expand recurrence rules into ``date/event/regime/importance`` rows.

    Dates are explicitly labeled fixture-quality in the result.  Reporting code
    should not present them as verified live release dates.
    """

    # Apply the same fail-closed schema for direct library callers as the main
    # YAML loader applies before a pipeline run.
    from ..config import _validate_catalysts_config

    _validate_catalysts_config(catalyst_config)

    start = pd.Timestamp(as_of).normalize()
    end = start + pd.Timedelta(days=int(horizon_days))
    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")
    adjustment = str(catalyst_config.get("business_day_adjustment", "following"))
    rows: list[dict[str, Any]] = []
    for rule in catalyst_config.get("recurring", []):
        cadence = rule.get("cadence")
        candidates: list[pd.Timestamp] = []
        if cadence == "monthly":
            month_starts = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
            configured_day = int(rule["day_of_month"])
            for month_start in month_starts:
                last_day = month_calendar.monthrange(month_start.year, month_start.month)[1]
                candidates.append(month_start.replace(day=min(configured_day, last_day)))
        elif cadence == "weekly":
            weekday = _WEEKDAYS[str(rule.get("weekday", "Monday")).lower()]
            candidates = [date for date in pd.date_range(start, end, freq="D") if date.weekday() == weekday]
            active_months = set(int(value) for value in rule.get("active_months", range(1, 13)))
            candidates = [date for date in candidates if date.month in active_months]
        elif cadence == "interval_days":
            interval = int(rule["interval_days"])
            if interval <= 0:
                raise ValueError("interval_days must be positive")
            anchor = pd.Timestamp(rule["anchor_date"]).normalize()
            while anchor < start:
                anchor += pd.Timedelta(days=interval)
            while anchor <= end:
                candidates.append(anchor)
                anchor += pd.Timedelta(days=interval)
        else:
            continue

        for candidate in candidates:
            if adjustment == "following":
                candidate = _following_business_day(candidate)
            if start <= candidate <= end:
                rows.append(
                    {
                        "date": candidate,
                        "event": str(rule["event"]),
                        "affected_regime": str(rule["affected_regime"]),
                        "importance": str(rule.get("expected_importance", "medium")),
                        "expected_importance": str(rule.get("expected_importance", "medium")),
                        "calendar_quality": "deterministic_fixture",
                    }
                )
    columns = [
        "date",
        "event",
        "affected_regime",
        "importance",
        "expected_importance",
        "calendar_quality",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(rows, columns=columns).sort_values(["date", "event"]).reset_index(drop=True)
