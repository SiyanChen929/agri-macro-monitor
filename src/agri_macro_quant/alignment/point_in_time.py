"""Vintage-aware, no-lookahead daily alignment.

The aligner treats every input row as a release event.  It replays those events
in chronological order and only emits a new state when the currently relevant
observation or its vintage changes.  This matters for revisions: a revision to
an old observation must not reset the age of a newer observation.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..data_models import AlignmentResult, SeriesFrequency, SeriesSpec, normalize_observation_frame
from ..validation.data_quality import DataQualityReport, validate_aligned_data, validate_observations


def _as_utc_naive(values: Any) -> Any:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Timestamp):
        return parsed.tz_localize(None)
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.tz_localize(None)
    return parsed.dt.tz_localize(None)


def _parse_decision_time(value: str | time) -> pd.Timedelta:
    if isinstance(value, time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    parts = str(value).split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("decision_time_utc must use HH:MM or HH:MM:SS")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
        raise ValueError("decision_time_utc is outside the valid clock range")
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


class PITAligner:
    """Align revision-aware observations to a canonical daily decision panel."""

    def __init__(
        self,
        specs: Mapping[str, SeriesSpec] | Iterable[SeriesSpec] | None = None,
        *,
        calendar_frequency: str = "B",
        decision_time_utc: str | time = "23:59:59",
        include_unconfigured: bool = True,
        quality_config: Mapping[str, Any] | None = None,
    ) -> None:
        if specs is None:
            self.specs: dict[str, SeriesSpec] = {}
        elif isinstance(specs, Mapping):
            self.specs = dict(specs)
        else:
            self.specs = {spec.series_id: spec for spec in specs}
        self.calendar_frequency = calendar_frequency
        self.decision_offset = _parse_decision_time(decision_time_utc)
        self.include_unconfigured = include_unconfigured
        self.quality_config = dict(quality_config or {})

    def _calendar(
        self,
        observations: pd.DataFrame,
        calendar: Iterable[Any] | pd.DatetimeIndex | None,
        start: Any | None,
        end: Any | None,
        as_of: Any | None,
    ) -> pd.DatetimeIndex:
        if calendar is not None:
            dates = pd.DatetimeIndex(_as_utc_naive(list(calendar))).normalize().drop_duplicates().sort_values()
        else:
            available = observations[
                ["publication_date", "revision_date", "effective_date"]
            ].max(axis=1)
            inferred_start = available.min() if start is None else _as_utc_naive(start)
            inferred_end = available.max() if end is None else _as_utc_naive(end)
            if pd.isna(inferred_start) or pd.isna(inferred_end):
                raise ValueError("Cannot infer an alignment calendar from empty or invalid observations")
            dates = pd.date_range(
                pd.Timestamp(inferred_start).normalize(),
                pd.Timestamp(inferred_end).normalize(),
                freq=self.calendar_frequency,
            )
        if start is not None:
            dates = dates[dates >= pd.Timestamp(_as_utc_naive(start)).normalize()]
        cap = pd.Timestamp(_as_utc_naive(as_of)).normalize() if as_of is not None else None
        if end is not None:
            end_cap = pd.Timestamp(_as_utc_naive(end)).normalize()
            cap = min(cap, end_cap) if cap is not None else end_cap
        if cap is not None:
            dates = dates[dates <= cap]
        if dates.empty:
            raise ValueError("Alignment calendar is empty after applying boundaries")
        dates.name = "date"
        return dates

    @staticmethod
    def _infer_spec(series_id: str, subset: pd.DataFrame) -> SeriesSpec:
        raw_frequency = str(subset["frequency"].dropna().iloc[-1]).lower() if subset["frequency"].notna().any() else "irregular"
        try:
            frequency = SeriesFrequency(raw_frequency)
        except ValueError:
            frequency = SeriesFrequency.IRREGULAR
        source = str(subset["source"].dropna().iloc[-1]) if subset["source"].notna().any() else "unknown"
        unit = str(subset["unit"].dropna().iloc[-1]) if subset["unit"].notna().any() else "index"
        return SeriesSpec(series_id=series_id, frequency=frequency, source=source, unit=unit)

    @staticmethod
    def _replay_releases(subset: pd.DataFrame) -> pd.DataFrame:
        """Build state-change events while preserving vintage semantics."""

        working = subset.copy()
        working["_available_at"] = working[
            ["publication_date", "revision_date", "effective_date"]
        ].max(axis=1)
        working = working.dropna(
            subset=["_available_at", "observation_date", "publication_date", "revision_date", "value"]
        )
        working = working[np.isfinite(working["value"].astype(float))]
        if working.empty:
            return pd.DataFrame(columns=["available_at", "value", "observation_date", "vintage_id"])
        working["_row_id"] = np.arange(len(working))
        working = working.sort_values(
            ["_available_at", "observation_date", "revision_date", "publication_date", "_row_id"]
        )

        vintages: dict[pd.Timestamp, pd.Series] = {}
        current_row_id: int | None = None
        events: list[dict[str, Any]] = []
        for available_at, release_batch in working.groupby("_available_at", sort=True):
            for _, row in release_batch.iterrows():
                observation_date = pd.Timestamp(row["observation_date"])
                previous = vintages.get(observation_date)
                candidate_key = (
                    pd.Timestamp(row["revision_date"]),
                    pd.Timestamp(row["publication_date"]),
                    int(row["_row_id"]),
                )
                if previous is None:
                    vintages[observation_date] = row
                else:
                    previous_key = (
                        pd.Timestamp(previous["revision_date"]),
                        pd.Timestamp(previous["publication_date"]),
                        int(previous["_row_id"]),
                    )
                    if candidate_key >= previous_key:
                        vintages[observation_date] = row

            latest_observation = max(vintages)
            selected = vintages[latest_observation]
            selected_id = int(selected["_row_id"])
            if selected_id != current_row_id:
                events.append(
                    {
                        "available_at": pd.Timestamp(available_at),
                        "value": float(selected["value"]),
                        "observation_date": pd.Timestamp(selected["observation_date"]),
                        "vintage_id": str(selected["vintage_id"]),
                    }
                )
                current_row_id = selected_id
        return pd.DataFrame.from_records(events).sort_values("available_at")

    def align(
        self,
        observations: pd.DataFrame,
        calendar: Iterable[Any] | pd.DatetimeIndex | None = None,
        *,
        start: Any | None = None,
        end: Any | None = None,
        as_of: Any | None = None,
    ) -> AlignmentResult:
        """Replay releases and return only information available at each date."""

        normalized = normalize_observation_frame(observations)
        dates = self._calendar(normalized, calendar, start, end, as_of)
        penalties = {
            "warning": float(self.quality_config.get("warning_penalty", 3.0)),
            "error": float(self.quality_config.get("error_penalty", 12.0)),
        }
        quality_args = {
            "warning_score": float(self.quality_config.get("warning_score", 85.0)),
            "failure_score": float(self.quality_config.get("failure_score", 60.0)),
            "min_critical_coverage": float(
                self.quality_config.get("min_critical_coverage", 0.80)
            ),
            "penalties": penalties,
        }
        raw_quality = validate_observations(
            normalized,
            self.specs,
            as_of=dates[-1],
            abnormal_jump_z=float(self.quality_config.get("abnormal_jump_z", 8.0)),
            stale_penalty=float(self.quality_config.get("stale_penalty", 4.0)),
            **quality_args,
        )
        decisions = pd.DataFrame(
            {
                "date": dates,
                "decision_at": dates + self.decision_offset,
            }
        )

        observed_names = set(normalized["series_id"].dropna().astype(str))
        names = list(self.specs)
        if self.include_unconfigured:
            names.extend(sorted(observed_names.difference(names)))
        if not names:
            names = sorted(observed_names)

        values: dict[str, pd.Series] = {}
        availability: dict[str, pd.Series] = {}
        observation_dates: dict[str, pd.Series] = {}
        ages: dict[str, pd.Series] = {}
        metadata_rows: list[dict[str, Any]] = []

        for series_id in names:
            subset = normalized[normalized["series_id"].astype(str) == series_id]
            spec = self.specs.get(series_id)
            if spec is None:
                spec = self._infer_spec(series_id, subset) if not subset.empty else SeriesSpec(
                    series_id=series_id,
                    frequency=SeriesFrequency.IRREGULAR,
                )
            actual_sources = sorted(subset["source"].dropna().astype(str).unique().tolist())
            actual_source = ",".join(actual_sources) if actual_sources else spec.source
            events = self._replay_releases(subset)
            if events.empty:
                values[series_id] = pd.Series(np.nan, index=dates, dtype=float)
                availability[series_id] = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
                observation_dates[series_id] = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
                ages[series_id] = pd.Series(np.nan, index=dates, dtype=float)
                metadata_rows.append(
                    {
                        "series_id": series_id,
                        "source": actual_source,
                        "configured_source": spec.source,
                        "frequency": spec.frequency.value,
                        "unit": spec.unit,
                        "critical": spec.critical,
                        "coverage": 0.0,
                        "last_available_date": pd.NaT,
                        "last_observation_date": pd.NaT,
                        "data_age_days": np.nan,
                        "max_staleness_days": spec.staleness_limit,
                        "stale": True,
                        "latest_value": np.nan,
                    }
                )
                continue

            merged = pd.merge_asof(
                decisions.sort_values("decision_at"),
                events.rename(columns={"available_at": "selected_available_at"}).sort_values("selected_available_at"),
                left_on="decision_at",
                right_on="selected_available_at",
                direction="backward",
                allow_exact_matches=True,
            ).set_index("date")
            age = (merged.index.to_series() - merged["selected_available_at"].dt.normalize()).dt.days.astype(float)
            stale = age > spec.staleness_limit
            aligned_value = merged["value"].where(~stale).astype(float)
            aligned_value.index = dates
            selected_available = merged["selected_available_at"]
            selected_available.index = dates
            selected_observation = merged["observation_date"]
            selected_observation.index = dates
            age.index = dates
            values[series_id] = aligned_value
            availability[series_id] = selected_available
            observation_dates[series_id] = selected_observation
            ages[series_id] = age

            final_age = age.iloc[-1]
            metadata_rows.append(
                {
                    "series_id": series_id,
                    "source": actual_source,
                    "configured_source": spec.source,
                    "frequency": spec.frequency.value,
                    "unit": spec.unit,
                    "critical": spec.critical,
                    "coverage": float(aligned_value.notna().mean()),
                    "last_available_date": selected_available.iloc[-1],
                    "last_observation_date": selected_observation.iloc[-1],
                    "data_age_days": final_age,
                    "max_staleness_days": spec.staleness_limit,
                    "stale": bool(pd.notna(final_age) and final_age > spec.staleness_limit),
                    "latest_value": aligned_value.iloc[-1],
                }
            )

        data_frame = pd.DataFrame(values, index=dates)
        availability_frame = pd.DataFrame(availability, index=dates)
        observation_frame = pd.DataFrame(observation_dates, index=dates)
        age_frame = pd.DataFrame(ages, index=dates)
        for frame in (data_frame, availability_frame, observation_frame, age_frame):
            frame.index.name = "date"
            frame.attrs.update(
                {
                    "point_in_time_aligned": True,
                    "decision_time_utc": str(self.decision_offset),
                    "source_mode": observations.attrs.get("mode", "unknown"),
                    "synthetic": bool(observations.attrs.get("synthetic", False)),
                }
            )
        aligned_quality = validate_aligned_data(
            data_frame,
            availability_frame,
            age_frame,
            self.specs,
            decision_at=dates + self.decision_offset,
            **quality_args,
        )
        quality = DataQualityReport.merge(
            raw_quality,
            aligned_quality,
            warning_score=quality_args["warning_score"],
            failure_score=quality_args["failure_score"],
            penalties=penalties,
        )
        metadata = pd.DataFrame(metadata_rows).set_index("series_id")
        return AlignmentResult(
            data=data_frame,
            quality=quality,
            metadata=metadata,
            availability=availability_frame,
            observation_dates=observation_frame,
            age_days=age_frame,
        )


def align_point_in_time(
    observations: pd.DataFrame,
    specs: Mapping[str, SeriesSpec] | Iterable[SeriesSpec] | None = None,
    calendar: Iterable[Any] | pd.DatetimeIndex | None = None,
    *,
    start: Any | None = None,
    end: Any | None = None,
    as_of: Any | None = None,
    decision_time_utc: str = "23:59:59",
    calendar_frequency: str = "B",
    quality_config: Mapping[str, Any] | None = None,
) -> AlignmentResult:
    """Functional wrapper around :class:`PITAligner`."""

    return PITAligner(
        specs,
        decision_time_utc=decision_time_utc,
        calendar_frequency=calendar_frequency,
        quality_config=quality_config,
    ).align(observations, calendar, start=start, end=end, as_of=as_of)
