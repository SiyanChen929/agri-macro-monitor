"""Orchestration for deterministic demo, cache, and optional live sources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, load_config
from ..data_models import ObservationFrame, empty_observation_frame, normalize_observation_frame
from .adapters import CSVAdapter, FREDAdapter, YahooChartAdapter
from .base import DataSourceError
from .demo import DemoAdapter
from .snapshot import infer_synthetic_provenance, load_snapshot


def _date_bounds(config: AppConfig, start: Any | None, end: Any | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start or config.runtime.start_date).normalize()
    configured_end = end if end is not None else config.runtime.end_date
    end_ts = (
        pd.Timestamp(configured_end).normalize()
        if configured_end is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    )
    if start_ts > end_ts:
        raise ValueError("Configured start_date must not be after end_date")
    return start_ts, end_ts


def _decision_cutoff(config: AppConfig, end: pd.Timestamp) -> pd.Timestamp:
    return end + pd.to_timedelta(config.runtime.decision_time_utc)


def load_demo_data(
    config: AppConfig | None = None,
    *,
    start: Any | None = None,
    end: Any | None = None,
    seed: int | None = None,
) -> ObservationFrame:
    """Return the complete deterministic offline fixture."""

    resolved = config or load_config()
    start_ts, end_ts = _date_bounds(resolved, start, end)
    adapter = DemoAdapter(resolved.series, seed=resolved.runtime.random_seed if seed is None else seed)
    return adapter.fetch(
        start_ts,
        end_ts,
        as_of=_decision_cutoff(resolved, end_ts),
    )


def _issue(code: str, message: str, series_id: str | None = None, severity: str = "error") -> dict[str, Any]:
    return {"code": code, "message": message, "series_id": series_id, "severity": severity}


def _fetch_live(config: AppConfig, start: pd.Timestamp, end: pd.Timestamp) -> ObservationFrame:
    frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []
    source_config = config.data_sources.get("sources", {})
    decision_as_of = _decision_cutoff(config, end)

    for series_id, spec in config.series.items():
        try:
            provider_key = "local_csv" if spec.source == "csv" else spec.source
            settings = source_config.get(provider_key, {})
            if isinstance(settings, dict) and not bool(settings.get("enabled", True)):
                issues.append(
                    _issue(
                        "source_disabled",
                        f"Configured source {spec.source!r} is disabled",
                        series_id,
                        "error" if spec.critical else "warning",
                    )
                )
                continue
            if (
                spec.source != "demo"
                and isinstance(settings, dict)
                and not bool(settings.get("live", True))
            ):
                issues.append(
                    _issue(
                        "source_not_live",
                        f"Configured source {spec.source!r} is not enabled for live runs",
                        series_id,
                        "error" if spec.critical else "warning",
                    )
                )
                continue
            if spec.source == "yahoo":
                adapter = YahooChartAdapter(
                    {series_id: spec.source_id or series_id},
                    {series_id: spec},
                    base_url=settings.get(
                        "base_url", "https://query1.finance.yahoo.com/v8/finance/chart"
                    ),
                    timeout=float(settings.get("timeout_seconds", 20)),
                    availability_time_utc=str(
                        settings.get("availability_time_utc", "23:00:00")
                    ),
                )
                frames.append(adapter.fetch(start, end, as_of=decision_as_of))
            elif spec.source == "fred":
                key_name = str(settings.get("api_key_env", "FRED_API_KEY"))
                adapter = FREDAdapter(
                    {series_id: spec.source_id or series_id},
                    {series_id: spec},
                    api_key=os.environ.get(key_name, ""),
                    base_url=settings.get(
                        "base_url", "https://api.stlouisfed.org/fred/series/observations"
                    ),
                    timeout=float(settings.get("timeout_seconds", 20)),
                    include_vintages=True,
                    availability_time_utc=str(
                        settings.get("availability_time_utc", "22:00:00")
                    ),
                )
                frames.append(adapter.fetch(start, end, as_of=decision_as_of))
            elif spec.source in {"csv", "local_csv"}:
                series_settings = config.data_sources.get("series", {}).get(series_id, {})
                location = (
                    series_settings.get("location")
                    or series_settings.get("path")
                    or spec.source_id
                )
                if not location:
                    raise DataSourceError(
                        "CSV series requires `location`, `path`, or `source_id`"
                    )
                expanded = os.path.expandvars(str(location))
                if "://" not in expanded:
                    candidate = Path(expanded).expanduser()
                    if not candidate.is_absolute():
                        candidate = config.project_root / candidate
                    expanded = str(candidate.resolve())
                adapter = CSVAdapter(
                    expanded,
                    series_id=series_id,
                    date_column=str(series_settings.get("date_column", "observation_date")),
                    value_column=str(series_settings.get("value_column", "value")),
                    spec=spec,
                )
                frames.append(adapter.fetch(start, end, as_of=decision_as_of))
            elif spec.source == "demo":
                issues.append(
                    _issue(
                        "live_source_not_configured",
                        "Series has only a demo feed and is omitted from live mode",
                        series_id,
                        "error" if spec.critical else "warning",
                    )
                )
            else:
                issues.append(
                    _issue(
                        "unsupported_live_source",
                        f"No live adapter is registered for source {spec.source!r}",
                        series_id,
                        "error" if spec.critical else "warning",
                    )
                )
        except DataSourceError as exc:
            issues.append(
                _issue(
                    "adapter_failure",
                    str(exc),
                    series_id,
                    "error" if spec.critical else "warning",
                )
            )

    result = pd.concat(frames, ignore_index=True, sort=False) if frames else empty_observation_frame()
    result = normalize_observation_frame(result)
    row_synthetic = infer_synthetic_provenance(result)
    if row_synthetic:
        issues.append(
            _issue(
                "synthetic_rows_in_live_mode",
                "A configured live/CSV feed returned rows labelled demo or synthetic",
                severity="error",
            )
        )
    result.attrs.update(
        {
            "mode": "live",
            "origin_mode": "mixed_with_synthetic" if row_synthetic else "live",
            "synthetic": row_synthetic,
            "ingestion_issues": issues,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
        }
    )
    return result


def ingest_all(
    config: AppConfig | None = None,
    mode: str | None = None,
    *,
    start: Any | None = None,
    end: Any | None = None,
    cache_path: str | Path | None = None,
) -> ObservationFrame:
    """Ingest every configured series without silently mixing live and demo data.

    ``demo`` is network-free.  ``live`` reports unavailable sources through
    ``frame.attrs['ingestion_issues']``.  ``cache`` loads an explicit prior
    canonical snapshot and never falls back to fabricated values.
    """

    resolved = config or load_config()
    selected_mode = mode or resolved.runtime.mode
    start_ts, end_ts = _date_bounds(resolved, start, end)
    if selected_mode == "demo":
        return load_demo_data(resolved, start=start_ts, end=end_ts)
    if selected_mode == "live":
        return _fetch_live(resolved, start_ts, end_ts)
    if selected_mode == "cache":
        location = Path(cache_path) if cache_path is not None else resolved.paths.cache_dir / "latest.csv.gz"
        source = load_snapshot(location)
        source_attrs = dict(source.attrs)
        decision_cutoff = _decision_cutoff(resolved, end_ts)
        available_at = source[
            ["publication_date", "effective_date", "revision_date"]
        ].max(axis=1)
        eligible = (
            available_at.le(decision_cutoff)
            & source["observation_date"].ge(start_ts)
            & source["observation_date"].le(end_ts)
        )
        result = normalize_observation_frame(source.loc[eligible].copy())
        result.attrs.update(source_attrs)
        result.attrs.update(
            {
                "mode": "cache",
                "source_snapshot_rows": int(len(source)),
                "excluded_post_cutoff_rows": int((~eligible).sum()),
                "requested_start": start_ts.isoformat(),
                "requested_end": end_ts.isoformat(),
                "decision_cutoff": decision_cutoff.isoformat(),
            }
        )
        return result
    raise ValueError("mode must be one of: demo, live, cache")
