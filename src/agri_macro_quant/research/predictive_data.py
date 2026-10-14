"""Public real-data acquisition for the exploratory grain prediction experiment.

The module deliberately keeps this data contract separate from the production
configuration.  Market prices are Yahoo continuous-futures research proxies;
USDA rows are the historical, as-published WASDE files; weather observations
come from NOAA NCEI Daily Summaries and are exposed only after a conservative
three-calendar-day publication lag.
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from ..ingestion.adapters import YahooChartAdapter


USDA_WASDE_URL = (
    "https://www.usda.gov/sites/default/files/documents/"
    "oce-wasde-report-data-{year:04d}-{month:02d}{suffix}.csv"
)
USDA_WASDE_ARCHIVES: tuple[dict[str, str], ...] = (
    {
        "start": "2010-04-01",
        "end": "2015-12-31",
        "url": (
            "https://www.usda.gov/sites/default/files/documents/"
            "oce-wasde-report-data-2010-04-to-2015-12.zip"
        ),
        # USDA's Akamai edge intermittently rejects automated ZIP requests.
        # This is an immutable raw replay of the exact official USDA URL.
        "fallback_url": (
            "https://web.archive.org/web/20220807051647id_/http://www.usda.gov/"
            "sites/default/files/documents/"
            "oce-wasde-report-data-2010-04-to-2015-12.zip"
        ),
        "sha256": "7eb285818ea748e15b562fe7aa88107999c42d8d6817f692d8f9818f16fddead",
        "member": "oce-wasde-report-data-2010-04-to-2015-12.csv",
        "output": "wasde_2010_04_to_2015_12.csv",
    },
    {
        "start": "2016-01-01",
        "end": "2020-12-31",
        "url": (
            "https://www.usda.gov/sites/default/files/documents/"
            "oce-wasde-report-data-2016-01-to-2020-12.zip"
        ),
        "fallback_url": (
            "https://web.archive.org/web/20220807050730id_/https://www.usda.gov/"
            "sites/default/files/documents/"
            "oce-wasde-report-data-2016-01-to-2020-12.zip"
        ),
        "sha256": "1cffbc4aa0ec436c832afb306b19631eaac42c2aaf3fffbf8dd9923052bfefb5",
        "member": "oce-wasde-report-data-2016-01-to-2020-12.csv",
        "output": "wasde_2016_01_to_2020_12.csv",
    },
)
USDA_REQUIRED_COLUMNS = {
    "WasdeNumber",
    "ReportDate",
    "ReportTitle",
    "Attribute",
    "Commodity",
    "Region",
    "MarketYear",
    "Value",
    "ReleaseDate",
    "ReleaseTime",
}
NOAA_DAILY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
USER_AGENT = "agri-macro-quant-research/0.1"

ASSET_SPECS: dict[str, dict[str, Any]] = {
    "ZW": {
        "symbol": "ZW=F",
        "commodity": "Wheat",
        "us_title": "U.S. Wheat Supply and Use",
        "world_title": "World Wheat Supply and Use",
    },
    "ZC": {
        "symbol": "ZC=F",
        "commodity": "Corn",
        "us_title": "U.S. Feed Grain and Corn Supply and Use",
        "world_title": "World Corn Supply and Use",
    },
    "ZS": {
        "symbol": "ZS=F",
        "commodity": "Oilseed, Soybean",
        "us_title": "U.S. Soybeans and Products Supply and Use (Domestic Measure)",
        "world_title": "World Soybean Supply and Use",
    },
}

NOAA_STATIONS: dict[str, tuple[str, ...]] = {
    # Winter-wheat Plains plus one Northern Plains spring-wheat station.
    "ZW": (
        "USW00003928",  # Wichita, Kansas
        "USW00013985",  # Dodge City, Kansas
        "USW00013967",  # Oklahoma City, Oklahoma
        "USW00023047",  # Amarillo, Texas
        "USW00014914",  # Fargo, North Dakota
    ),
    # A deliberately fixed central Corn Belt basket, shared by corn/soybeans.
    "ZC": (
        "USW00014933",  # Des Moines, Iowa
        "USW00014942",  # Omaha, Nebraska
        "USW00003945",  # Columbia, Missouri
        "USW00093819",  # Indianapolis, Indiana
        "USW00014922",  # Minneapolis, Minnesota
        "USW00014944",  # Sioux Falls, South Dakota
    ),
    "ZS": (
        "USW00014933",
        "USW00014942",
        "USW00003945",
        "USW00093819",
        "USW00014922",
        "USW00014944",
    ),
}


@dataclass(frozen=True)
class PredictiveDataBundle:
    """Paths and lineage for one downloaded experiment dataset."""

    market_prices: pd.DataFrame
    usda_features: pd.DataFrame
    weather_features: pd.DataFrame
    lineage: pd.DataFrame
    data_dir: Path


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str,
    *,
    timeout: float = 90.0,
    attempts: int = 3,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Referer": "https://www.usda.gov/historical-wasde-report-data-3",
                "Accept": (
                    "text/csv,application/zip,application/json;q=0.9,*/*;q=0.8"
                ),
            },
        )
        try:
            with urlopen(  # noqa: S310 - fixed public endpoints
                request, timeout=timeout
            ) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(0.4 * (2**attempt))
    raise RuntimeError(
        f"download failed for {url.split('?')[0]}: {last_error}"
    ) from last_error


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[int, int]]:
    first = max(pd.Timestamp("2010-04-01"), start.to_period("M").to_timestamp())
    for period in pd.period_range(first, end, freq="M"):
        yield int(period.year), int(period.month)


def _wasde_header(payload: bytes) -> set[str]:
    """Read a WASDE CSV header without trusting a response content type."""

    first_line = payload[:8192].decode("utf-8-sig", errors="replace").splitlines()
    if not first_line:
        return set()
    return {str(value).strip() for value in next(csv.reader([first_line[0]]), [])}


def _valid_wasde_csv_payload(payload: bytes) -> bool:
    return USDA_REQUIRED_COLUMNS.issubset(_wasde_header(payload))


def _valid_wasde_csv_path(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return _valid_wasde_csv_payload(handle.read(8192))
    except OSError:
        return False


def _extract_wasde_archive(payload: bytes, spec: Mapping[str, str], target: Path) -> None:
    """Validate a pinned USDA ZIP and atomically extract its sole CSV member."""

    if not payload.startswith(b"PK"):
        raise ValueError("response was not a ZIP archive")
    actual_hash = sha256(payload).hexdigest()
    if actual_hash != spec["sha256"]:
        raise ValueError(
            f"archive SHA-256 mismatch: expected {spec['sha256']}, got {actual_hash}"
        )
    try:
        with ZipFile(BytesIO(payload)) as archive:
            names = archive.namelist()
            if names != [spec["member"]]:
                raise ValueError(f"unexpected archive members: {names!r}")
            csv_payload = archive.read(spec["member"])
    except BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {exc}") from exc
    if not _valid_wasde_csv_payload(csv_payload):
        raise ValueError("archive CSV did not contain the expected WASDE header")
    _atomic_write(target, csv_payload)


def _wasde_release_dates(files: Iterable[Path]) -> pd.DatetimeIndex:
    """Collect distinct report releases without loading the large archives at once."""

    release_dates: set[pd.Timestamp] = set()
    for path in files:
        try:
            chunks = pd.read_csv(
                path,
                usecols=["ReleaseDate"],
                chunksize=250_000,
                low_memory=False,
            )
            for chunk in chunks:
                parsed = pd.to_datetime(chunk["ReleaseDate"], errors="coerce").dropna()
                release_dates.update(
                    pd.Timestamp(value).normalize() for value in parsed.unique()
                )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"could not validate WASDE releases in {path}: {exc}"
            ) from exc
    return pd.DatetimeIndex(sorted(release_dates))


def download_usda_wasde(
    destination: str | Path,
    *,
    start: Any,
    end: Any,
    refresh: bool = False,
    workers: int = 4,
) -> tuple[list[Path], list[str]]:
    """Download official as-published WASDE CSVs and return files/failures.

    USDA distributes April 2010--December 2020 as two consolidated ZIPs and
    2021 onward as monthly CSVs.  The ZIP URLs are occasionally rejected by
    USDA's Akamai edge, so the downloader first tries USDA and then an
    immutable Internet Archive replay of the *same official URL*, pinned by
    SHA-256.  Monthly files remain direct USDA downloads.
    """

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    start_date = max(pd.Timestamp("2010-04-01"), pd.Timestamp(start).normalize())
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("WASDE end date must not precede April 2010/start")
    failures: list[str] = []
    files: list[Path] = []

    for spec in USDA_WASDE_ARCHIVES:
        archive_start = pd.Timestamp(spec["start"])
        archive_end = pd.Timestamp(spec["end"])
        if archive_end < start_date or archive_start > end_date:
            continue
        path = root / spec["output"]
        if path.exists() and not refresh and _valid_wasde_csv_path(path):
            files.append(path)
            continue
        archive_errors: list[str] = []
        for source in (spec["url"], spec["fallback_url"]):
            try:
                payload = _download(source, timeout=180.0, attempts=2)
                _extract_wasde_archive(payload, spec, path)
                files.append(path)
                break
            except (RuntimeError, ValueError) as exc:
                archive_errors.append(f"{source.split('/')[2]}: {exc}")
        else:
            failures.append(
                f"{archive_start:%Y-%m}--{archive_end:%Y-%m}: "
                + " | ".join(archive_errors)
            )

    monthly_start = max(start_date, pd.Timestamp("2021-01-01"))
    months = (
        list(_month_range(monthly_start, end_date)) if end_date >= monthly_start else []
    )

    def fetch_one(item: tuple[int, int]) -> Path | None:
        year, month = item
        path = root / f"wasde_{year:04d}_{month:02d}.csv"
        if path.exists() and not refresh and _valid_wasde_csv_path(path):
            return path
        # Corrected USDA files occasionally carry a V2 suffix (for example,
        # May and June 2026); probe the official naming variants explicitly.
        for suffix in ("", "-V2", "-v2", "_V2", "_v2"):
            url = USDA_WASDE_URL.format(year=year, month=month, suffix=suffix)
            try:
                payload = _download(url, attempts=1)
            except RuntimeError:
                continue
            # Akamai sometimes returns a small HTML Access Denied page with 200.
            if not _valid_wasde_csv_payload(payload):
                continue
            _atomic_write(path, payload)
            return path
        return None

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(fetch_one, item): item for item in months}
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                path = future.result()
            except Exception as exc:  # defensive: preserve the complete failure ledger
                failures.append(f"{year:04d}-{month:02d}: {exc}")
                continue
            if path is None:
                failures.append(f"{year:04d}-{month:02d}: unavailable")
            else:
                files.append(path)

    files = sorted(set(files))
    if not files:
        raise RuntimeError("No historical WASDE files were available")

    release_dates = _wasde_release_dates(files)
    covered = release_dates[(release_dates >= start_date) & (release_dates <= end_date)]
    requested_months = pd.period_range(start_date, end_date, freq="M")
    observed_months = {value.to_period("M") for value in covered}
    missing_months = [period for period in requested_months if period not in observed_months]
    existing_failure_months = {failure[:7] for failure in failures}
    for period in missing_months:
        key = str(period)
        if key not in existing_failure_months:
            failures.append(f"{key}: no release found in acquired official files")

    # Allow a small number of genuinely absent monthly reports (for example,
    # shutdown months), but reject a silently truncated archive.
    allowed_missing = max(3, int(np.ceil(len(requested_months) * 0.04)))
    minimum_releases = max(1, len(requested_months) - allowed_missing)
    if len(observed_months) < minimum_releases:
        raise RuntimeError(
            f"Only {len(observed_months)} of {len(requested_months)} requested WASDE report "
            f"months were available; expected at least {minimum_releases}"
        )
    return files, sorted(failures)


def download_noaa_weather(
    path: str | Path,
    *,
    start: Any,
    end: Any,
    refresh: bool = False,
) -> Path:
    """Download the preregistered crop-region station basket from NOAA NCEI."""

    target = Path(path)
    if target.exists() and not refresh:
        return target
    stations = sorted({station for values in NOAA_STATIONS.values() for station in values})
    query = urlencode(
        {
            "dataset": "daily-summaries",
            "stations": ",".join(stations),
            "startDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "endDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "dataTypes": "PRCP,TMAX,TMIN,TAVG",
            "format": "csv",
            "units": "metric",
            "includeStationName": "true",
            "includeStationLocation": "true",
        }
    )
    payload = _download(f"{NOAA_DAILY_URL}?{query}", timeout=180.0)
    if b"STATION" not in payload[:512] or b"DATE" not in payload[:1024]:
        raise RuntimeError("NOAA response did not contain the expected daily-summaries CSV")
    _atomic_write(target, payload)
    return target


def download_market_prices(
    path: str | Path,
    *,
    start: Any,
    end: Any,
    refresh: bool = False,
) -> Path:
    """Download Yahoo continuous-futures proxies and persist a wide price frame."""

    target = Path(path)
    if target.exists() and not refresh:
        return target
    adapter = YahooChartAdapter(
        {asset: spec["symbol"] for asset, spec in ASSET_SPECS.items()},
        availability_time_utc="23:00:00",
    )
    observations = adapter.fetch(
        start,
        end,
        as_of=pd.Timestamp(end).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59),
    )
    prices = observations.pivot_table(
        index="observation_date", columns="series_id", values="value", aggfunc="last"
    ).sort_index()
    prices.index.name = "date"
    if prices.reindex(columns=ASSET_SPECS).count().min() < 1000:
        raise RuntimeError("Yahoo returned insufficient history for one or more grain proxies")
    target.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(target, compression="gzip")
    return target


def _market_year_start(value: Any) -> float:
    token = str(value).split("/")[0].strip()
    try:
        return float(int(token))
    except (TypeError, ValueError):
        return np.nan


def build_usda_features(files: Iterable[str | Path]) -> pd.DataFrame:
    """Parse report-vintage USDA rows into commodity-specific release features."""

    selected: list[pd.DataFrame] = []
    use_columns = [
        "ReportTitle",
        "Attribute",
        "Commodity",
        "Region",
        "MarketYear",
        "Value",
        "ReleaseDate",
        "ReleaseTime",
    ]
    required_attributes = {
        "Production",
        "Yield per Harvested Acre",
        "Ending Stocks",
        "Use, Total",
        "Domestic Total",
    }
    for raw_path in files:
        try:
            frame = pd.read_csv(raw_path, usecols=use_columns, low_memory=False)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame["market_year_start"] = frame["MarketYear"].map(_market_year_start)
        for asset, spec in ASSET_SPECS.items():
            commodity = str(spec["commodity"])
            for region_key, region, title in (
                ("us", "United States", str(spec["us_title"])),
                ("world", "World", str(spec["world_title"])),
            ):
                mask = (
                    frame["Commodity"].eq(commodity)
                    & frame["Region"].eq(region)
                    & frame["ReportTitle"].eq(title)
                    & frame["Attribute"].isin(required_attributes)
                    & frame["Value"].notna()
                    & frame["market_year_start"].notna()
                )
                subset = frame.loc[mask].copy()
                if subset.empty:
                    continue
                subset["asset"] = asset
                subset["region_key"] = region_key
                selected.append(subset)

    if not selected:
        raise RuntimeError("No usable commodity rows were parsed from the WASDE archive")
    rows = pd.concat(selected, ignore_index=True)
    # USDA changed the CSV date representation during 2021 (MM/DD/YYYY to
    # ISO-8601).  Pandas' strict vector parser otherwise infers one format and
    # silently drops valid releases written in the other representation.
    try:
        rows["release_date"] = pd.to_datetime(
            rows["ReleaseDate"], errors="coerce", format="mixed"
        )
    except (TypeError, ValueError):  # pandas < 2.0 compatibility
        iso_dates = pd.to_datetime(
            rows["ReleaseDate"], errors="coerce", format="%Y-%m-%d"
        )
        us_dates = pd.to_datetime(
            rows["ReleaseDate"], errors="coerce", format="%m/%d/%Y"
        )
        rows["release_date"] = iso_dates.fillna(us_dates)
    rows = rows.dropna(subset=["release_date", "asset", "Value"])
    # Some historical files contain duplicated presentation rows. The report,
    # commodity, region, market-year, and attribute identity is unique after
    # taking the last published value in that monthly CSV.
    grouped = (
        rows.sort_values("release_date")
        .groupby(
            ["release_date", "asset", "region_key", "market_year_start", "Attribute"],
            as_index=False,
        )["Value"]
        .last()
    )
    wide = grouped.pivot_table(
        index=["release_date", "asset", "market_year_start"],
        columns=["region_key", "Attribute"],
        values="Value",
        aggfunc="last",
    ).sort_index()
    wide.columns = [
        f"{region}_{str(attribute).lower().replace(',', '').replace(' ', '_').replace('.', '')}"
        for region, attribute in wide.columns
    ]
    wide = wide.reset_index()

    # The 2010--2020 ZIPs each contain many report vintages, and a report can
    # mention an incomplete future crop year in one table before the matching
    # U.S./world balance sheets exist.  Select the most recent *complete* crop
    # year separately for every release and asset.  Doing this per input file
    # silently collapsed each consolidated ZIP to its final crop year.
    selection_columns = [
        "us_production",
        "us_yield_per_harvested_acre",
        "us_ending_stocks",
        "us_use_total",
        "us_domestic_total",
        "world_production",
        "world_ending_stocks",
    ]
    available_selection_columns = [
        column for column in selection_columns if column in wide.columns
    ]
    wide["_coverage"] = wide[available_selection_columns].notna().sum(axis=1)
    best_coverage = wide.groupby(["release_date", "asset"])["_coverage"].transform(
        "max"
    )
    wide = wide.loc[wide["_coverage"].eq(best_coverage)]
    latest_complete_year = wide.groupby(["release_date", "asset"])[
        "market_year_start"
    ].transform("max")
    wide = (
        wide.loc[wide["market_year_start"].eq(latest_complete_year)]
        .drop(columns="_coverage")
        .sort_values(["asset", "market_year_start", "release_date"])
    )

    expected = {
        "us_production": "us_production_revision",
        "us_yield_per_harvested_acre": "us_yield_revision",
        "us_ending_stocks": "us_ending_stocks_revision",
        "world_production": "world_production_revision",
        "world_ending_stocks": "world_ending_stocks_revision",
    }
    group_keys = [wide["asset"], wide["market_year_start"]]
    for source, target in expected.items():
        values = pd.to_numeric(wide.get(source), errors="coerce")
        wide[target] = values.groupby(group_keys).pct_change(fill_method=None)
    use = pd.to_numeric(wide.get("us_use_total"), errors="coerce")
    if "us_domestic_total" in wide:
        use = use.combine_first(
            pd.to_numeric(wide["us_domestic_total"], errors="coerce")
        )
    ending = pd.to_numeric(wide.get("us_ending_stocks"), errors="coerce")
    wide["us_stocks_to_use"] = ending / use.replace(0.0, np.nan)
    wide["us_stocks_to_use_revision"] = wide["us_stocks_to_use"].groupby(group_keys).diff()
    # USDA publishes the consolidated historical CSV on the following day.
    # Activating it on report day would assume access to the noon XML/PDF,
    # which this CSV-only experiment has not independently reconstructed.
    wide["available_date"] = wide["release_date"] + pd.Timedelta(days=1)
    columns = [
        "available_date",
        "release_date",
        "asset",
        "market_year_start",
        *expected.values(),
        "us_stocks_to_use",
        "us_stocks_to_use_revision",
    ]
    result = wide.reindex(columns=columns).sort_values(["asset", "available_date"])
    return result.reset_index(drop=True)


def _seasonal_prior_mean(values: pd.Series, month_day: pd.Series) -> pd.Series:
    scratch = pd.DataFrame({"value": values, "month_day": month_day})
    return scratch.groupby("month_day", sort=False)["value"].transform(
        lambda series: series.expanding(min_periods=3).mean().shift(1)
    )


def build_weather_features(path: str | Path, *, publication_lag_days: int = 3) -> pd.DataFrame:
    """Build causal crop-basket weather anomalies from NOAA daily summaries."""

    raw = pd.read_csv(
        path,
        usecols=["STATION", "DATE", "PRCP", "TMAX", "TMIN", "TAVG"],
        low_memory=False,
    )
    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
    for column in ("PRCP", "TMAX", "TMIN", "TAVG"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["TAVG"] = raw["TAVG"].fillna((raw["TMAX"] + raw["TMIN"]) / 2.0)
    raw = raw.dropna(subset=["DATE", "STATION"])

    panels: list[pd.DataFrame] = []
    for asset, stations in NOAA_STATIONS.items():
        subset = raw.loc[raw["STATION"].isin(stations)].copy()
        daily = subset.groupby("DATE", sort=True).agg(
            temperature=("TAVG", "mean"),
            maximum_temperature=("TMAX", "mean"),
            precipitation=("PRCP", "mean"),
            station_count=("STATION", "nunique"),
        )
        daily = daily.sort_index()
        month_day = pd.Series(daily.index.strftime("%m-%d"), index=daily.index)
        temperature_normal = _seasonal_prior_mean(daily["temperature"], month_day)
        precipitation_normal = _seasonal_prior_mean(daily["precipitation"], month_day)
        temperature_anomaly = daily["temperature"] - temperature_normal
        precipitation_anomaly = daily["precipitation"] - precipitation_normal
        output = pd.DataFrame(index=daily.index)
        output["temperature_anomaly_14d"] = temperature_anomaly.rolling(
            14, min_periods=7
        ).mean()
        output["precipitation_anomaly_30d"] = precipitation_anomaly.rolling(
            30, min_periods=15
        ).sum()
        output["heat_stress_days_14d"] = daily["maximum_temperature"].ge(35.0).rolling(
            14, min_periods=7
        ).sum()
        growing_months = {3, 4, 5, 6, 7} if asset == "ZW" else {4, 5, 6, 7, 8, 9}
        growing = pd.Series(daily.index.month, index=daily.index).isin(growing_months).astype(float)
        output["growing_season_temperature_anomaly"] = (
            output["temperature_anomaly_14d"] * growing
        )
        output["growing_season_precipitation_anomaly"] = (
            output["precipitation_anomaly_30d"] * growing
        )
        output["weather_station_count"] = daily["station_count"]
        output["observation_date"] = output.index
        output["available_date"] = output.index + pd.Timedelta(days=publication_lag_days)
        output["asset"] = asset
        panels.append(output.reset_index(drop=True))
    return pd.concat(panels, ignore_index=True).sort_values(["asset", "available_date"])


def acquire_predictive_data(
    data_dir: str | Path,
    *,
    start: Any,
    end: Any,
    refresh: bool = False,
) -> PredictiveDataBundle:
    """Acquire, parse, cache, and hash all public inputs for the experiment."""

    root = Path(data_dir).resolve()
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    market_path = download_market_prices(
        raw / "yahoo_grain_continuous.csv.gz", start=start, end=end, refresh=refresh
    )
    noaa_path = download_noaa_weather(
        raw / "noaa_crop_belt_daily.csv", start=start, end=end, refresh=refresh
    )
    usda_files, usda_failures = download_usda_wasde(
        raw / "usda_wasde", start=start, end=end, refresh=refresh
    )

    prices = pd.read_csv(market_path, index_col="date", parse_dates=True)
    prices = prices.reindex(columns=ASSET_SPECS).sort_index()
    usda = build_usda_features(usda_files)
    weather = build_weather_features(noaa_path)
    usda_path = processed / "usda_release_features.csv.gz"
    weather_path = processed / "weather_features.csv.gz"
    usda.to_csv(usda_path, index=False, compression="gzip")
    weather.to_csv(weather_path, index=False, compression="gzip")

    lineage_rows = [
        {
            "dataset": "market_prices",
            "provider": "Yahoo chart",
            "quality": "vendor_continuous_proxy_not_contract_level",
            "path": str(market_path.relative_to(root)),
            "sha256": _sha256_path(market_path),
            "rows": int(len(prices)),
            "first_date": prices.index.min().isoformat(),
            "last_date": prices.index.max().isoformat(),
            "failures": "",
        },
        {
            "dataset": "weather",
            "provider": "NOAA NCEI Daily Summaries",
            "quality": "official_observations_conservative_3d_lag",
            "path": str(noaa_path.relative_to(root)),
            "sha256": _sha256_path(noaa_path),
            "rows": int(len(weather)),
            "first_date": pd.to_datetime(weather["observation_date"]).min().isoformat(),
            "last_date": pd.to_datetime(weather["observation_date"]).max().isoformat(),
            "failures": "",
        },
        {
            "dataset": "usda_wasde",
            "provider": "USDA WAOB historical WASDE; pinned archive replay fallback",
            "quality": "official_as_published_csvs_validated_headers_and_hashes",
            "path": str((raw / "usda_wasde").relative_to(root)),
            "sha256": sha256(
                "".join(_sha256_path(path) for path in usda_files).encode("utf-8")
            ).hexdigest(),
            "rows": int(len(usda)),
            "first_date": pd.to_datetime(usda["release_date"]).min().isoformat(),
            "last_date": pd.to_datetime(usda["release_date"]).max().isoformat(),
            "failures": "; ".join(usda_failures),
        },
    ]
    lineage = pd.DataFrame(lineage_rows)
    lineage.to_csv(processed / "data_lineage.csv", index=False)
    return PredictiveDataBundle(
        market_prices=prices,
        usda_features=usda,
        weather_features=weather,
        lineage=lineage,
        data_dir=root,
    )


__all__ = [
    "ASSET_SPECS",
    "NOAA_STATIONS",
    "PredictiveDataBundle",
    "acquire_predictive_data",
    "build_usda_features",
    "build_weather_features",
    "download_market_prices",
    "download_noaa_weather",
    "download_usda_wasde",
]
