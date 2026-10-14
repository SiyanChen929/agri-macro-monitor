"""Optional public live adapters and local canonical CSV ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..data_models import ObservationFrame, SeriesSpec, normalize_observation_frame
from .base import BaseAdapter, DataSourceError, FetchContext


def _clock_offset(value: str) -> pd.Timedelta:
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError("availability time must use HH:MM or HH:MM:SS")
    hours, minutes = parts[:2]
    seconds = parts[2] if len(parts) == 3 else 0
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
        raise ValueError("availability time is outside the valid UTC clock range")
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _read_url_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "agri-macro-quant/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured public endpoint
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(f"HTTP request failed for {url.split('?')[0]}: {exc}") from exc
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataSourceError("Remote endpoint returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise DataSourceError("Remote endpoint returned a non-object JSON payload")
    return result


class CSVAdapter(BaseAdapter):
    """Load a local/HTTP canonical long CSV or one simple value series."""

    source_name = "csv"

    def __init__(
        self,
        location: str | Path,
        *,
        series_id: str | None = None,
        date_column: str = "observation_date",
        value_column: str = "value",
        spec: SeriesSpec | None = None,
    ) -> None:
        self.location = str(location)
        self.series_id = series_id
        self.date_column = date_column
        self.value_column = value_column
        self.spec = spec

    def fetch(self, start: Any, end: Any, *, as_of: Any | None = None) -> ObservationFrame:
        context = FetchContext.create(start, end, as_of)
        try:
            frame = pd.read_csv(self.location)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataSourceError(f"Could not read CSV source {self.location}: {exc}") from exc
        if "series_id" in frame.columns and self.series_id is not None:
            frame = frame.loc[
                frame["series_id"].astype(str).eq(str(self.series_id))
            ].copy()
            if frame.empty:
                raise DataSourceError(
                    f"Canonical CSV {self.location} has no rows for {self.series_id!r}"
                )
        if "series_id" not in frame.columns:
            if self.series_id is None:
                raise DataSourceError("A non-canonical CSV requires series_id")
            if self.date_column not in frame or self.value_column not in frame:
                raise DataSourceError(
                    f"CSV requires {self.date_column!r} and {self.value_column!r} columns"
                )
            observation_date = pd.to_datetime(frame[self.date_column], errors="coerce")
            lag = self.spec.publication_lag_days if self.spec else 0
            publication_date = (
                observation_date
                + pd.to_timedelta(lag, unit="D")
                + pd.Timedelta(hours=23, minutes=59, seconds=59)
            )
            frame = pd.DataFrame(
                {
                    "series_id": self.series_id,
                    "observation_date": observation_date,
                    "publication_date": publication_date,
                    "effective_date": observation_date,
                    "revision_date": publication_date,
                    "value": pd.to_numeric(frame[self.value_column], errors="coerce"),
                    "source": self.source_name,
                    "frequency": self.spec.frequency.value if self.spec else "irregular",
                    "unit": self.spec.unit if self.spec else "index",
                    "vintage_id": "initial",
                    "timestamp_quality": "configured_lag",
                }
            )
        result = normalize_observation_frame(frame)
        mask = (
            result["observation_date"].between(context.start, context.end)
            & (result[["publication_date", "revision_date", "effective_date"]].max(axis=1) <= context.as_of)
        )
        return result.loc[mask].reset_index(drop=True)


class YahooChartAdapter(BaseAdapter):
    """Yahoo chart endpoint adapter for daily market proxies.

    Yahoo continuous futures are convenient public research proxies, not
    exchange-quality back-adjusted contracts.  ``roll_method`` is tagged in the
    output so production users cannot mistake them for a validated roll series.
    """

    source_name = "yahoo"

    def __init__(
        self,
        symbols: Mapping[str, str],
        specs: Mapping[str, SeriesSpec] | None = None,
        *,
        base_url: str = "https://query1.finance.yahoo.com/v8/finance/chart",
        timeout: float = 20.0,
        availability_time_utc: str = "23:00:00",
    ) -> None:
        self.symbols = dict(symbols)
        self.specs = dict(specs or {})
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.availability_offset = _clock_offset(availability_time_utc)

    def fetch(self, start: Any, end: Any, *, as_of: Any | None = None) -> ObservationFrame:
        context = FetchContext.create(start, end, as_of)
        rows: list[dict[str, Any]] = []
        for series_id, ticker in self.symbols.items():
            period1 = int(context.start.tz_localize("UTC").timestamp()) if context.start.tzinfo is None else int(context.start.timestamp())
            inclusive_end = context.end + pd.Timedelta(days=1)
            period2 = int(inclusive_end.tz_localize("UTC").timestamp()) if inclusive_end.tzinfo is None else int(inclusive_end.timestamp())
            query = urlencode(
                {
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "history",
                    "includeAdjustedClose": "true",
                }
            )
            payload = _read_url_json(f"{self.base_url}/{quote(ticker, safe='')}?{query}", self.timeout)
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise DataSourceError(f"Yahoo error for {ticker}: {chart['error']}")
            results = chart.get("result") or []
            if not results:
                raise DataSourceError(f"Yahoo returned no observations for {ticker}")
            result = results[0]
            timestamps = result.get("timestamp") or []
            indicators = result.get("indicators", {})
            adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
            closes = (indicators.get("quote") or [{}])[0].get("close")
            values = adjusted or closes or []
            spec = self.specs.get(series_id)
            for unix_time, value in zip(timestamps, values, strict=False):
                if value is None or not np.isfinite(float(value)):
                    continue
                vendor_timestamp = pd.Timestamp(
                    unix_time, unit="s", tz="UTC"
                ).tz_localize(None)
                observation = vendor_timestamp.normalize()
                release = max(
                    vendor_timestamp,
                    observation + self.availability_offset,
                )
                if release > context.as_of:
                    continue
                rows.append(
                    {
                        "series_id": series_id,
                        "observation_date": observation,
                        "publication_date": release,
                        "effective_date": observation,
                        "revision_date": release,
                        "value": float(value) * (spec.price_scale if spec else 1.0),
                        "source": self.source_name,
                        "frequency": "daily",
                        "unit": spec.unit if spec else "price",
                        "vintage_id": "initial",
                        "source_id": ticker,
                        "vendor_bar_timestamp": vendor_timestamp,
                        "source_value_scale": spec.price_scale if spec else 1.0,
                        "timestamp_quality": "configured_conservative_close_after_vendor_bar",
                        "roll_method": "vendor_continuous_proxy" if ticker.endswith("=F") else "not_applicable",
                    }
                )
        if not rows:
            raise DataSourceError("Yahoo returned no usable values")
        return normalize_observation_frame(pd.DataFrame.from_records(rows))


class FREDAdapter(BaseAdapter):
    """FRED/ALFRED adapter supporting revision vintages when requested."""

    source_name = "fred"

    def __init__(
        self,
        series: Mapping[str, str],
        specs: Mapping[str, SeriesSpec] | None = None,
        *,
        api_key: str,
        base_url: str = "https://api.stlouisfed.org/fred/series/observations",
        timeout: float = 20.0,
        include_vintages: bool = True,
        availability_time_utc: str = "22:00:00",
    ) -> None:
        if not api_key:
            raise DataSourceError("FRED_API_KEY is required for the live FRED adapter")
        self.series = dict(series)
        self.specs = dict(specs or {})
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.include_vintages = include_vintages
        self.availability_offset = _clock_offset(availability_time_utc)

    def _fetch_one(self, series_id: str, fred_id: str, context: FetchContext) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "series_id": fred_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": context.start.strftime("%Y-%m-%d"),
            "observation_end": context.end.strftime("%Y-%m-%d"),
            "limit": 100000,
            "sort_order": "asc",
        }
        if self.include_vintages:
            params.update(
                {
                    "realtime_start": context.start.strftime("%Y-%m-%d"),
                    "realtime_end": context.as_of.strftime("%Y-%m-%d"),
                    "output_type": 3,
                }
            )
        payload = _read_url_json(f"{self.base_url}?{urlencode(params)}", self.timeout)
        observations = payload.get("observations")
        if not isinstance(observations, list):
            message = payload.get("error_message", "missing observations")
            raise DataSourceError(f"FRED error for {fred_id}: {message}")
        spec = self.specs.get(series_id)
        parsed: list[dict[str, Any]] = []
        for item in observations:
            raw_value = item.get("value")
            if raw_value in {None, "."}:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            observation_date = pd.Timestamp(item["date"])
            if self.include_vintages and item.get("realtime_start"):
                revision_date = (
                    pd.Timestamp(item["realtime_start"]).normalize()
                    + self.availability_offset
                )
                timestamp_quality = "alfred_vintage_conservative_release_time"
            else:
                lag = spec.publication_lag_days if spec else 1
                revision_date = (
                    observation_date
                    + pd.Timedelta(days=lag)
                    + self.availability_offset
                )
                timestamp_quality = "approximated_configured_lag_final_value"
            parsed.append(
                {
                    "series_id": series_id,
                    "observation_date": observation_date,
                    "_revision_date": revision_date,
                    "value": value,
                    "source": self.source_name,
                    "frequency": spec.frequency.value if spec else "irregular",
                    "unit": spec.unit if spec else "index",
                    "source_id": fred_id,
                    "timestamp_quality": timestamp_quality,
                }
            )
        if not parsed:
            return []
        first_release: dict[pd.Timestamp, pd.Timestamp] = {}
        for row in parsed:
            date = row["observation_date"]
            first_release[date] = min(first_release.get(date, row["_revision_date"]), row["_revision_date"])
        rows: list[dict[str, Any]] = []
        for row in parsed:
            revision = row.pop("_revision_date")
            publication = first_release[row["observation_date"]]
            row.update(
                {
                    "publication_date": publication,
                    "effective_date": row["observation_date"],
                    "revision_date": revision,
                    "vintage_id": revision.strftime("%Y-%m-%d"),
                }
            )
            rows.append(row)
        return rows

    def fetch(self, start: Any, end: Any, *, as_of: Any | None = None) -> ObservationFrame:
        context = FetchContext.create(start, end, as_of)
        rows: list[dict[str, Any]] = []
        for series_id, fred_id in self.series.items():
            rows.extend(self._fetch_one(series_id, fred_id, context))
        if not rows:
            raise DataSourceError("FRED returned no usable observations")
        return normalize_observation_frame(pd.DataFrame.from_records(rows))
