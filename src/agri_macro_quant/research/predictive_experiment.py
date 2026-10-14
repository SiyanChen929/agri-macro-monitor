"""Leakage-controlled agricultural return-prediction experiment.

This module deliberately stops at statistical prediction.  It accepts an
already point-in-time-aligned market panel plus optional, already parsed USDA
and weather feature panels.  It neither downloads data nor interprets a
synthetic fixture as empirical evidence.

The prediction target is the futures price return earned from the next session
through the configured horizon.  Trading costs are not subtracted from this
label because realized costs depend on the forecast sign and the prior
position; callers should evaluate the returned forecasts with the package's
path-aware backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


ASSET_PRICE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ZW": ("ZW", "wheat", "ZW=F"),
    "ZC": ("ZC", "corn", "ZC=F"),
    "ZS": ("ZS", "soybeans", "soybean", "ZS=F"),
}

CROP_YEAR_START_MONTH: Mapping[str, int] = {
    "ZW": 6,
    "ZC": 9,
    "ZS": 9,
}

PROXY_CONTRACT_MONTHS: Mapping[str, tuple[int, ...]] = {
    "ZW": (3, 5, 7, 9, 12),
    "ZC": (3, 5, 7, 9, 12),
    "ZS": (1, 3, 5, 7, 8, 9, 11),
}

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "core__annual_sin",
    "core__annual_cos",
    "core__semiannual_sin",
    "core__semiannual_cos",
    "core__crop_year_fraction",
    "core__momentum_5",
    "core__momentum_20",
    "core__momentum_60",
    "core__momentum_120",
    "core__realized_volatility_20",
    "core__realized_volatility_60",
    "core__ma_distance_20",
    "core__ma_distance_60",
    "core__cross_grain_momentum_20",
    "core__cross_grain_momentum_60",
    "core__relative_momentum_20",
    "core__relative_momentum_60",
)

BASELINE_MODELS: tuple[str, ...] = (
    "historical_mean",
    "seasonality",
    "momentum_60",
)

RIDGE_CORE = "ridge_core"
RIDGE_CORE_USDA = "ridge_core_usda"
RIDGE_CORE_WEATHER = "ridge_core_weather"
RIDGE_CORE_BOTH = "ridge_core_usda_weather"
RIDGE_CANDIDATE_ORDER: tuple[str, ...] = (
    RIDGE_CORE_USDA,
    RIDGE_CORE_WEATHER,
    RIDGE_CORE_BOTH,
)


@dataclass(frozen=True)
class PredictiveExperimentConfig:
    """Immutable specification for the predeclared experiment."""

    assets: tuple[str, ...] = ("ZW", "ZC", "ZS")
    horizon_sessions: int = 21
    weekly_rule: str = "W-FRI"
    minimum_train_crop_years: int = 5
    embargo_business_days: int = 5
    holdout_crop_years: int = 3
    minimum_weekly_observations_per_crop_year: int = 40
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    minimum_feature_coverage: float = 0.60
    neutralize_proxy_roll_gaps: bool = True
    proxy_roll_anchor_day: int = 15
    proxy_roll_maximum_day: int = 21
    alphas: tuple[float, ...] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
    )

    def __post_init__(self) -> None:
        assets = tuple(str(asset).upper() for asset in self.assets)
        if not assets:
            raise ValueError("assets cannot be empty")
        unknown = sorted(set(assets).difference(CROP_YEAR_START_MONTH))
        if unknown:
            raise ValueError(f"unsupported crop-year asset(s): {unknown}")
        object.__setattr__(self, "assets", assets)
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if self.minimum_train_crop_years < 1:
            raise ValueError("minimum_train_crop_years must be positive")
        if self.embargo_business_days < 0:
            raise ValueError("embargo_business_days cannot be negative")
        if self.holdout_crop_years < 1:
            raise ValueError("holdout_crop_years must be positive")
        if self.minimum_weekly_observations_per_crop_year < 1:
            raise ValueError(
                "minimum_weekly_observations_per_crop_year must be positive"
            )
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 1.0:
            raise ValueError("winsor bounds must satisfy 0 <= lower < upper <= 1")
        if not 0.0 < self.minimum_feature_coverage <= 1.0:
            raise ValueError("minimum_feature_coverage must lie in (0, 1]")
        if not 1 <= int(self.proxy_roll_anchor_day) <= 28:
            raise ValueError("proxy_roll_anchor_day must lie in [1, 28]")
        if not int(self.proxy_roll_anchor_day) <= int(self.proxy_roll_maximum_day) <= 31:
            raise ValueError(
                "proxy_roll_maximum_day must be between anchor day and 31"
            )
        alphas = tuple(float(alpha) for alpha in self.alphas)
        if not alphas or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alphas):
            raise ValueError("alphas must be a non-empty sequence of finite non-negative values")
        object.__setattr__(self, "alphas", alphas)


@dataclass(frozen=True)
class CropYearFold:
    """One expanding outer fold represented by immutable date tuples."""

    asset: str
    test_crop_year: int
    train_crop_years: tuple[int, ...]
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    is_holdout: bool
    purge_cutoff: pd.Timestamp


@dataclass(frozen=True)
class PredictiveExperimentResult:
    """Immutable references to all experiment tables and selections."""

    config: PredictiveExperimentConfig
    proxy_input: bool
    selected_model: str
    feature_groups: Mapping[str, tuple[str, ...]]
    holdout_crop_years: Mapping[str, tuple[int, ...]]
    daily_panel: pd.DataFrame
    weekly_panel: pd.DataFrame
    folds: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    hyperparameters: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_asset: pd.DataFrame
    metrics_by_crop_year: pd.DataFrame
    development_selected_model: str | None = None


def _datetime_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    if "date" in result.columns and not isinstance(result.index, pd.DatetimeIndex):
        result = result.set_index("date")
    if isinstance(result.index, pd.MultiIndex):
        raise TypeError(f"{name} must have a one-dimensional date index")
    result.index = pd.to_datetime(result.index, errors="raise")
    if result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _resolve_prices(
    market_prices: pd.DataFrame,
    assets: Sequence[str],
) -> pd.DataFrame:
    frame = _datetime_frame(market_prices, "market_prices")
    output: dict[str, pd.Series] = {}
    lower_lookup = {str(column).lower(): column for column in frame.columns}
    for asset in assets:
        column: Any | None = None
        for alias in ASSET_PRICE_ALIASES[str(asset)]:
            if alias in frame.columns:
                column = alias
                break
            column = lower_lookup.get(alias.lower())
            if column is not None:
                break
        if column is None:
            raise KeyError(
                f"market_prices has no column for {asset}; accepted aliases are "
                f"{ASSET_PRICE_ALIASES[str(asset)]}"
            )
        numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
        output[str(asset)] = numeric.where(numeric > 0.0)
    result = pd.DataFrame(output, index=frame.index).sort_index()
    result.index.name = "date"
    return result


def _neutralize_proxy_roll_gaps(
    prices: pd.DataFrame,
    config: PredictiveExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove only predeclared generic-futures stitch jumps from proxy returns.

    The first valid session on or after the configured anchor day in every
    listed delivery month is neutralized.  Yahoo does not expose the held
    contract identity, so this deterministic calendar rule prevents a
    cross-contract price ratio from becoming a return without using future
    magnitude information.  It necessarily removes that session's genuine
    same-contract move too; this remains a research proxy, not exchange-grade
    roll P&L.
    """

    adjusted = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for asset in prices.columns:
        raw = pd.to_numeric(prices[asset], errors="coerce").dropna()
        if raw.empty:
            continue
        returns = raw.pct_change(fill_method=None)
        date = pd.DatetimeIndex(raw.index)
        eligible_window = (
            date.month.isin(PROXY_CONTRACT_MONTHS[str(asset)])
            & (date.day >= int(config.proxy_roll_anchor_day))
            & (date.day <= int(config.proxy_roll_maximum_day))
        )
        eligible = pd.Series(eligible_window, index=raw.index, dtype=bool)
        delivery_group = pd.MultiIndex.from_arrays(
            [date.year, date.month], names=["year", "month"]
        )
        first_eligible = eligible.groupby(delivery_group).cumsum().eq(1)
        stitch = eligible & first_eligible
        repaired_returns = returns.mask(stitch, 0.0).fillna(0.0)
        reconstructed = float(raw.iloc[0]) * (1.0 + repaired_returns).cumprod()
        adjusted.loc[raw.index, asset] = reconstructed.to_numpy(dtype=float)
        flags.loc[raw.index, asset] = stitch.to_numpy(dtype=bool)
    return adjusted.astype(float), flags.astype(bool)


def asset_crop_year(dates: Any, asset: str) -> pd.Index | int:
    """Return the crop-year start year for dates belonging to ``asset``."""

    code = str(asset).upper()
    if code not in CROP_YEAR_START_MONTH:
        raise ValueError(f"unsupported crop-year asset: {asset!r}")
    parsed = pd.to_datetime(dates)
    scalar = isinstance(parsed, pd.Timestamp)
    index = pd.DatetimeIndex([parsed] if scalar else parsed)
    start_month = CROP_YEAR_START_MONTH[code]
    values = np.where(index.month >= start_month, index.year, index.year - 1).astype(int)
    return int(values[0]) if scalar else pd.Index(values, dtype="int64", name="crop_year")


def _crop_year_bounds(asset: str, crop_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    month = CROP_YEAR_START_MONTH[str(asset).upper()]
    start = pd.Timestamp(year=int(crop_year), month=month, day=1)
    end = pd.Timestamp(year=int(crop_year) + 1, month=month, day=1) - pd.Timedelta(days=1)
    return start, end


def _feature_frame_long(
    frame: pd.DataFrame | None,
    *,
    prefix: str,
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalize a global or asset-specific feature frame to date/asset rows."""

    empty_index = pd.MultiIndex.from_product(
        [dates, assets], names=["date", "asset"]
    )
    if frame is None or frame.empty:
        return pd.DataFrame(index=empty_index), ()

    raw = frame.copy()
    if {"date", "asset"}.issubset(raw.columns):
        raw["date"] = pd.to_datetime(raw["date"], errors="raise")
        raw["asset"] = raw["asset"].astype(str).str.upper()
        raw = raw.set_index(["date", "asset"])
    elif isinstance(raw.index, pd.MultiIndex):
        if raw.index.nlevels != 2:
            raise TypeError(f"{prefix} feature index must have date and asset levels")
        reset = raw.reset_index()
        date_column, asset_column = reset.columns[:2]
        reset[date_column] = pd.to_datetime(reset[date_column], errors="raise")
        reset[asset_column] = reset[asset_column].astype(str).str.upper()
        raw = reset.set_index([date_column, asset_column])
        raw.index.names = ["date", "asset"]
    else:
        global_frame = _datetime_frame(raw, f"{prefix}_features")
        pieces: list[pd.DataFrame] = []
        for asset in assets:
            piece = global_frame.copy()
            piece["asset"] = str(asset)
            piece = piece.set_index("asset", append=True)
            piece.index.names = ["date", "asset"]
            pieces.append(piece)
        raw = pd.concat(pieces).sort_index()

    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    rename = {
        column: str(column)
        if str(column).startswith(f"{prefix}__")
        else f"{prefix}__{column}"
        for column in raw.columns
    }
    raw = raw.rename(columns=rename).reindex(empty_index)
    usable = tuple(str(column) for column in raw if raw[column].notna().any())
    return raw, usable


def _forward_target(
    price: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return next-session-entry labels and their entry/exit dates.

    A forecast made at decision session ``t`` enters at the following valid
    asset session and exits ``horizon`` valid sessions after that entry.  Thus
    a 21-session label uses prices at ``t + 1`` and ``t + 22``.  Shifting on
    the non-missing asset series makes the convention robust to holidays and
    isolated missing marks.
    """

    valid = pd.to_numeric(price, errors="coerce").where(lambda x: x > 0.0).dropna()
    entry_price = valid.shift(-1)
    exit_price = valid.shift(-(int(horizon) + 1))
    target = exit_price / entry_price - 1.0
    date_values = pd.Series(valid.index, index=valid.index)
    start_values = date_values.shift(-1)
    end_values = date_values.shift(-(int(horizon) + 1))
    target_full = target.reindex(price.index).astype(float)
    start_full = pd.to_datetime(start_values.reindex(price.index), errors="coerce")
    end_full = pd.to_datetime(end_values.reindex(price.index), errors="coerce")
    return target_full, start_full, end_full


def build_daily_prediction_panel(
    market_prices: pd.DataFrame,
    usda_features: pd.DataFrame | None = None,
    weather_features: pd.DataFrame | None = None,
    config: PredictiveExperimentConfig | None = None,
    *,
    proxy_input: bool = False,
) -> tuple[pd.DataFrame, Mapping[str, tuple[str, ...]]]:
    """Build the causal daily long panel and feature-family dictionary.

    USDA and weather inputs must already obey their publication/vintage lags.
    This function joins them exactly as supplied and never forward-fills them.
    """

    cfg = config or PredictiveExperimentConfig()
    raw_prices = _resolve_prices(market_prices, cfg.assets)
    if bool(proxy_input):
        repaired_prices, roll_gap_flags = _neutralize_proxy_roll_gaps(raw_prices, cfg)
        prices = repaired_prices if cfg.neutralize_proxy_roll_gaps else raw_prices.copy()
    else:
        prices = raw_prices.copy()
        roll_gap_flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    momentum: dict[int, pd.DataFrame] = {
        window: prices / prices.shift(window) - 1.0
        for window in (5, 20, 60, 120)
    }
    daily_returns = prices.pct_change(fill_method=None)
    volatility = {
        window: daily_returns.rolling(window, min_periods=window).std(ddof=0)
        * np.sqrt(252.0)
        for window in (20, 60)
    }
    moving_average = {
        window: prices / prices.rolling(window, min_periods=window).mean() - 1.0
        for window in (20, 60)
    }
    cross_momentum = {
        window: values.mean(axis=1, skipna=False)
        for window, values in momentum.items()
        if window in {20, 60}
    }

    pieces: list[pd.DataFrame] = []
    for asset in cfg.assets:
        index = prices.index
        day_angle = 2.0 * np.pi * index.dayofyear.to_numpy(dtype=float) / 365.25
        crop_year = asset_crop_year(index, asset)
        start_dates = pd.DatetimeIndex(
            [
                _crop_year_bounds(asset, int(year))[0]
                for year in crop_year.to_numpy(dtype=int)
            ]
        )
        end_dates = pd.DatetimeIndex(
            [
                _crop_year_bounds(asset, int(year))[1]
                for year in crop_year.to_numpy(dtype=int)
            ]
        )
        elapsed = (index - start_dates).days.to_numpy(dtype=float)
        duration = (end_dates - start_dates).days.to_numpy(dtype=float) + 1.0
        target, target_start, target_end = _forward_target(
            prices[asset], cfg.horizon_sessions
        )
        raw_target, raw_target_start, raw_target_end = _forward_target(
            raw_prices[asset], cfg.horizon_sessions
        )
        frame = pd.DataFrame(index=index)
        frame["asset"] = asset
        frame["price"] = prices[asset]
        frame["raw_price"] = raw_prices[asset]
        frame["proxy_roll_gap_flag"] = roll_gap_flags[asset]
        frame["target_return_21d"] = target
        frame["target_start_date"] = target_start
        frame["target_end_date"] = target_end
        frame["raw_target_return_21d"] = raw_target
        frame["raw_target_start_date"] = raw_target_start
        frame["raw_target_end_date"] = raw_target_end
        frame["crop_year"] = crop_year.to_numpy(dtype=int)
        frame["seasonality_bucket"] = (
            (index.month - CROP_YEAR_START_MONTH[asset]) % 12
        ).astype(int)
        frame["core__annual_sin"] = np.sin(day_angle)
        frame["core__annual_cos"] = np.cos(day_angle)
        frame["core__semiannual_sin"] = np.sin(2.0 * day_angle)
        frame["core__semiannual_cos"] = np.cos(2.0 * day_angle)
        frame["core__crop_year_fraction"] = elapsed / duration
        for window in (5, 20, 60, 120):
            frame[f"core__momentum_{window}"] = momentum[window][asset]
        for window in (20, 60):
            frame[f"core__realized_volatility_{window}"] = volatility[window][asset]
            frame[f"core__ma_distance_{window}"] = moving_average[window][asset]
            frame[f"core__cross_grain_momentum_{window}"] = cross_momentum[window]
            frame[f"core__relative_momentum_{window}"] = (
                momentum[window][asset] - cross_momentum[window]
            )
        frame = frame.set_index("asset", append=True)
        frame.index.names = ["date", "asset"]
        pieces.append(frame)

    panel = pd.concat(pieces).sort_index()
    usda, usda_columns = _feature_frame_long(
        usda_features,
        prefix="usda",
        assets=cfg.assets,
        dates=prices.index,
    )
    weather, weather_columns = _feature_frame_long(
        weather_features,
        prefix="weather",
        assets=cfg.assets,
        dates=prices.index,
    )
    panel = panel.join(usda, how="left").join(weather, how="left")
    groups = {
        "core": tuple(column for column in CORE_FEATURE_COLUMNS if column in panel),
        "usda": usda_columns,
        "weather": weather_columns,
    }
    panel.attrs["point_in_time_feature_inputs_required"] = True
    panel.attrs["target_horizon_sessions"] = cfg.horizon_sessions
    panel.attrs["proxy_input"] = bool(proxy_input)
    panel.attrs["proxy_roll_gap_neutralized"] = bool(
        proxy_input and cfg.neutralize_proxy_roll_gaps
    )
    return panel, groups


def sample_weekly_panel(
    daily_panel: pd.DataFrame,
    config: PredictiveExperimentConfig | None = None,
) -> pd.DataFrame:
    """Select the last priced observation in each configured week."""

    cfg = config or PredictiveExperimentConfig()
    if not isinstance(daily_panel.index, pd.MultiIndex) or daily_panel.index.nlevels != 2:
        raise TypeError("daily_panel must use a (date, asset) MultiIndex")
    selected: list[pd.DataFrame] = []
    assets_present = set(daily_panel.index.get_level_values("asset").astype(str))
    for asset in cfg.assets:
        if asset not in assets_present:
            continue
        frame = daily_panel.xs(asset, level="asset").sort_index()
        frame = frame.loc[frame["price"].notna()]
        rows: list[pd.DataFrame] = []
        for _, group in frame.groupby(pd.Grouper(freq=cfg.weekly_rule)):
            if not group.empty:
                rows.append(group.iloc[[-1]])
        if rows:
            sampled = pd.concat(rows).sort_index()
            sampled["asset"] = asset
            sampled = sampled.set_index("asset", append=True)
            sampled.index.names = ["date", "asset"]
            selected.append(sampled)
    if not selected:
        raise ValueError("weekly sampling produced no observations")
    result = pd.concat(selected).sort_index()
    result.attrs.update(daily_panel.attrs)
    result.attrs["weekly_rule"] = cfg.weekly_rule
    return result


def _complete_crop_years(
    frame: pd.DataFrame,
    asset: str,
    config: PredictiveExperimentConfig,
) -> tuple[int, ...]:
    complete: list[int] = []
    for crop_year, group in frame.groupby("crop_year", sort=True):
        start, end = _crop_year_bounds(asset, int(crop_year))
        if len(group) < config.minimum_weekly_observations_per_crop_year:
            continue
        dates = pd.DatetimeIndex(group.index)
        covers_start = dates.min() <= start + pd.Timedelta(days=10)
        covers_end = dates.max() >= end - pd.Timedelta(days=10)
        targets_complete = group["target_return_21d"].notna().all()
        end_dates_complete = group["target_end_date"].notna().all()
        if covers_start and covers_end and targets_complete and end_dates_complete:
            complete.append(int(crop_year))
    return tuple(sorted(complete))


def make_crop_year_folds(
    weekly_panel: pd.DataFrame,
    asset: str,
    config: PredictiveExperimentConfig | None = None,
) -> tuple[CropYearFold, ...]:
    """Create expanding folds with target-overlap purge and an embargo."""

    cfg = config or PredictiveExperimentConfig()
    code = str(asset).upper()
    if code not in cfg.assets:
        raise ValueError(f"asset {code!r} is outside the configured universe")
    if not isinstance(weekly_panel.index, pd.MultiIndex):
        raise TypeError("weekly_panel must use a (date, asset) MultiIndex")
    try:
        frame = weekly_panel.xs(code, level="asset").sort_index()
    except KeyError as exc:
        raise ValueError(f"weekly_panel has no rows for {code}") from exc
    frame.index = pd.to_datetime(frame.index)
    complete = _complete_crop_years(frame, code, cfg)
    required = cfg.minimum_train_crop_years + cfg.holdout_crop_years + 1
    if len(complete) < required:
        raise ValueError(
            f"{code} has {len(complete)} complete crop years; at least {required} "
            "are required for development selection and the holdout"
        )
    holdout = set(complete[-cfg.holdout_crop_years :])
    folds: list[CropYearFold] = []
    for position in range(cfg.minimum_train_crop_years, len(complete)):
        test_year = int(complete[position])
        train_years = tuple(int(year) for year in complete[:position])
        test = frame.loc[frame["crop_year"].astype(int).eq(test_year)]
        test = test.loc[test["target_return_21d"].notna()]
        if test.empty:
            continue
        purge_cutoff = pd.Timestamp(test.index.min()) - BDay(
            cfg.embargo_business_days
        )
        train = frame.loc[frame["crop_year"].astype(int).isin(train_years)]
        target_end = pd.to_datetime(train["target_end_date"], errors="coerce")
        train = train.loc[
            train["target_return_21d"].notna()
            & target_end.notna()
            & target_end.lt(purge_cutoff)
        ]
        retained_years = tuple(
            sorted(train["crop_year"].dropna().astype(int).unique().tolist())
        )
        if len(retained_years) < cfg.minimum_train_crop_years:
            continue
        folds.append(
            CropYearFold(
                asset=code,
                test_crop_year=test_year,
                train_crop_years=retained_years,
                train_dates=tuple(pd.Timestamp(date) for date in train.index),
                test_dates=tuple(pd.Timestamp(date) for date in test.index),
                is_holdout=test_year in holdout,
                purge_cutoff=pd.Timestamp(purge_cutoff),
            )
        )
    if not any(not fold.is_holdout for fold in folds):
        raise ValueError(f"{code} has no development folds before the holdout")
    if len([fold for fold in folds if fold.is_holdout]) != cfg.holdout_crop_years:
        raise ValueError(f"{code} does not have the requested complete holdout folds")
    return tuple(folds)


@dataclass(frozen=True)
class _PreparedFeatures:
    train: np.ndarray
    test: np.ndarray
    names: tuple[str, ...]


def _prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    config: PredictiveExperimentConfig,
) -> _PreparedFeatures:
    train_numeric = train.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    test_numeric = test.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    coverage = train_numeric.notna().mean()
    selected = list(coverage.index[coverage >= config.minimum_feature_coverage])
    if not selected:
        raise ValueError("no features meet the training-only coverage threshold")
    train_numeric = train_numeric[selected]
    test_numeric = test_numeric[selected]
    lower = train_numeric.quantile(config.winsor_lower)
    upper = train_numeric.quantile(config.winsor_upper)
    train_clipped = train_numeric.clip(lower=lower, upper=upper, axis=1)
    test_clipped = test_numeric.clip(lower=lower, upper=upper, axis=1)
    medians = train_clipped.median()
    train_missing = train_clipped.isna().astype(float)
    test_missing = test_clipped.isna().astype(float)
    train_filled = train_clipped.fillna(medians)
    test_filled = test_clipped.fillna(medians)

    combined_train = pd.concat(
        [
            train_filled,
            train_missing.rename(columns=lambda value: f"{value}__missing"),
        ],
        axis=1,
    )
    combined_test = pd.concat(
        [
            test_filled,
            test_missing.rename(columns=lambda value: f"{value}__missing"),
        ],
        axis=1,
    )
    means = combined_train.mean()
    scales = combined_train.std(ddof=0)
    varying = scales.gt(1.0e-12) & scales.map(np.isfinite)
    combined_train = combined_train.loc[:, varying]
    combined_test = combined_test.loc[:, varying]
    means = means.loc[varying]
    scales = scales.loc[varying]
    if combined_train.shape[1] == 0:
        raise ValueError("all training features are constant")
    train_array = ((combined_train - means) / scales).to_numpy(dtype=float)
    test_array = ((combined_test - means) / scales).to_numpy(dtype=float)
    return _PreparedFeatures(
        train=train_array,
        test=test_array,
        names=tuple(map(str, combined_train.columns)),
    )


def _ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    alpha: float,
    config: PredictiveExperimentConfig,
) -> tuple[np.ndarray, float, np.ndarray, tuple[str, ...]]:
    prepared = _prepare_features(train, test, columns, config)
    y = pd.to_numeric(train["target_return_21d"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("ridge training target contains missing or infinite values")
    intercept = float(y.mean())
    centred = y - intercept
    gram = prepared.train.T @ prepared.train
    rhs = prepared.train.T @ centred
    penalty = float(alpha) * np.eye(gram.shape[0])
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + penalty) @ rhs
    prediction = intercept + prepared.test @ coefficients
    return prediction.astype(float), intercept, coefficients.astype(float), prepared.names


def _select_alpha(
    train: pd.DataFrame,
    columns: Sequence[str],
    config: PredictiveExperimentConfig,
) -> tuple[float, list[dict[str, Any]]]:
    years = sorted(train["crop_year"].dropna().astype(int).unique().tolist())
    records: list[dict[str, Any]] = []
    if len(years) < 2:
        selected = min(config.alphas, key=lambda value: abs(np.log10(max(value, 1.0e-12))))
        return float(selected), records
    validation_year = int(years[-1])
    validation = train.loc[train["crop_year"].astype(int).eq(validation_year)]
    if validation.empty:
        return float(config.alphas[0]), records
    cutoff = pd.Timestamp(validation.index.min()) - BDay(config.embargo_business_days)
    inner_train = train.loc[train["crop_year"].astype(int).lt(validation_year)]
    target_end = pd.to_datetime(inner_train["target_end_date"], errors="coerce")
    inner_train = inner_train.loc[target_end.notna() & target_end.lt(cutoff)]
    if inner_train.empty:
        return float(config.alphas[0]), records
    actual = pd.to_numeric(validation["target_return_21d"], errors="coerce").to_numpy(dtype=float)
    for alpha in config.alphas:
        try:
            prediction, _, _, _ = _ridge_fit_predict(
                inner_train, validation, columns, float(alpha), config
            )
            mse = float(np.mean((actual - prediction) ** 2))
        except ValueError:
            mse = np.inf
        records.append(
            {
                "inner_validation_crop_year": validation_year,
                "alpha": float(alpha),
                "inner_validation_mse": mse,
                "inner_train_observations": int(len(inner_train)),
                "inner_validation_observations": int(len(validation)),
            }
        )
    finite = [record for record in records if np.isfinite(record["inner_validation_mse"])]
    selected = (
        min(finite, key=lambda record: (record["inner_validation_mse"], record["alpha"]))[
            "alpha"
        ]
        if finite
        else config.alphas[0]
    )
    for record in records:
        record["selected"] = bool(float(record["alpha"]) == float(selected))
    return float(selected), records


def _model_feature_columns(
    feature_groups: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    core = tuple(feature_groups.get("core", ()))
    usda = tuple(feature_groups.get("usda", ()))
    weather = tuple(feature_groups.get("weather", ()))
    output: dict[str, tuple[str, ...]] = {RIDGE_CORE: core}
    if usda:
        output[RIDGE_CORE_USDA] = core + usda
    if weather:
        output[RIDGE_CORE_WEATHER] = core + weather
    if usda and weather:
        output[RIDGE_CORE_BOTH] = core + usda + weather
    return output


def _fold_baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Mapping[str, np.ndarray]:
    target = pd.to_numeric(train["target_return_21d"], errors="coerce")
    historical_mean = float(target.mean())
    seasonal_means = target.groupby(train["seasonality_bucket"].astype(int)).mean()
    seasonal = (
        test["seasonality_bucket"].astype(int).map(seasonal_means).fillna(historical_mean)
    )
    momentum = pd.to_numeric(test["core__momentum_60"], errors="coerce").fillna(0.0)
    return {
        "historical_mean": np.full(len(test), historical_mean, dtype=float),
        "seasonality": seasonal.to_numpy(dtype=float),
        "momentum_60": momentum.to_numpy(dtype=float),
    }


def _prediction_records(
    *,
    asset: str,
    fold: CropYearFold,
    split: str,
    test: pd.DataFrame,
    model: str,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    actual = pd.to_numeric(test["target_return_21d"], errors="coerce")
    for position, date in enumerate(test.index):
        records.append(
            {
                "date": pd.Timestamp(date),
                "asset": asset,
                "crop_year": int(fold.test_crop_year),
                "split": split,
                "model": model,
                "actual": float(actual.iloc[position]),
                "prediction": float(prediction[position]),
                "target_start_date": pd.Timestamp(
                    test["target_start_date"].iloc[position]
                ),
                "target_end_date": pd.Timestamp(test["target_end_date"].iloc[position]),
            }
        )
    return records


def _run_fold_models(
    frame: pd.DataFrame,
    fold: CropYearFold,
    models: Mapping[str, tuple[str, ...]],
    config: PredictiveExperimentConfig,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = frame.reindex(pd.DatetimeIndex(fold.train_dates)).copy()
    test = frame.reindex(pd.DatetimeIndex(fold.test_dates)).copy()
    train = train.loc[train["target_return_21d"].notna()]
    test = test.loc[test["target_return_21d"].notna()]
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    for model, values in _fold_baseline_predictions(train, test).items():
        predictions.extend(
            _prediction_records(
                asset=fold.asset,
                fold=fold,
                split=split,
                test=test,
                model=model,
                prediction=values,
            )
        )
    for model, columns in models.items():
        alpha, alpha_records = _select_alpha(train, columns, config)
        prediction, intercept, values, names = _ridge_fit_predict(
            train, test, columns, alpha, config
        )
        predictions.extend(
            _prediction_records(
                asset=fold.asset,
                fold=fold,
                split=split,
                test=test,
                model=model,
                prediction=prediction,
            )
        )
        coefficients.append(
            {
                "asset": fold.asset,
                "crop_year": fold.test_crop_year,
                "split": split,
                "model": model,
                "alpha": alpha,
                "feature": "__intercept__",
                "coefficient_standardized": intercept,
            }
        )
        coefficients.extend(
            {
                "asset": fold.asset,
                "crop_year": fold.test_crop_year,
                "split": split,
                "model": model,
                "alpha": alpha,
                "feature": name,
                "coefficient_standardized": float(value),
            }
            for name, value in zip(names, values, strict=True)
        )
        for record in alpha_records:
            hyperparameters.append(
                {
                    "asset": fold.asset,
                    "crop_year": fold.test_crop_year,
                    "split": split,
                    "model": model,
                    **record,
                }
            )
        if not alpha_records:
            hyperparameters.append(
                {
                    "asset": fold.asset,
                    "crop_year": fold.test_crop_year,
                    "split": split,
                    "model": model,
                    "inner_validation_crop_year": np.nan,
                    "alpha": alpha,
                    "inner_validation_mse": np.nan,
                    "inner_train_observations": 0,
                    "inner_validation_observations": 0,
                    "selected": True,
                }
            )
    return predictions, coefficients, hyperparameters


def _metric_row(frame: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce")
    valid = actual.notna() & prediction.notna() & np.isfinite(actual) & np.isfinite(prediction)
    actual = actual.loc[valid]
    prediction = prediction.loc[valid]
    if actual.empty:
        return {
            "observations": 0.0,
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_ic": np.nan,
            "spearman_ic": np.nan,
            "directional_accuracy": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    error = actual - prediction
    has_variation = (
        len(actual) > 1
        and actual.nunique(dropna=True) > 1
        and prediction.nunique(dropna=True) > 1
    )
    pearson = (
        float(actual.corr(prediction, method="pearson"))
        if has_variation
        else np.nan
    )
    spearman = (
        float(actual.corr(prediction, method="spearman"))
        if has_variation
        else np.nan
    )
    directional = float((np.sign(actual) == np.sign(prediction)).mean())
    design = np.column_stack([np.ones(len(prediction)), prediction.to_numpy(dtype=float)])
    calibration = np.linalg.lstsq(design, actual.to_numpy(dtype=float), rcond=None)[0]
    return {
        "observations": float(len(actual)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_ic": pearson,
        "spearman_ic": spearman,
        "directional_accuracy": directional,
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def _metrics_table(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(group_columns), sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, key_tuple, strict=True)),
                **_metric_row(group),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    mean_lookup = (
        result.loc[result["model"].eq("historical_mean")]
        .set_index([column for column in group_columns if column != "model"])["mse"]
    )
    baseline_keys = [column for column in group_columns if column != "model"]
    if baseline_keys:
        lookup_index = (
            pd.MultiIndex.from_frame(result[baseline_keys])
            if len(baseline_keys) > 1
            else pd.Index(result[baseline_keys[0]])
        )
        baseline_mse = pd.Series(
            [mean_lookup.get(key, np.nan) for key in lookup_index], index=result.index
        )
    else:
        baseline_value = float(mean_lookup.iloc[0]) if len(mean_lookup) else np.nan
        baseline_mse = pd.Series(baseline_value, index=result.index)
    result["oos_r2_vs_historical_mean"] = 1.0 - result["mse"] / baseline_mse
    return result.sort_values(list(group_columns)).reset_index(drop=True)


def _grouped_model_mse(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.Series:
    """Compute model MSEs on finite, development-only forecast rows."""

    frame = predictions.copy()
    if "split" in frame:
        frame = frame.loc[frame["split"].eq("development")]
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    forecast = pd.to_numeric(frame["prediction"], errors="coerce")
    valid = actual.notna() & forecast.notna() & np.isfinite(actual) & np.isfinite(forecast)
    frame = frame.loc[valid].copy()
    frame["squared_error"] = (
        actual.loc[valid].to_numpy(dtype=float)
        - forecast.loc[valid].to_numpy(dtype=float)
    ) ** 2
    keys = ["model", *group_columns]
    return frame.groupby(keys, sort=True, dropna=False)["squared_error"].mean()


def _select_development_model(
    development_predictions: pd.DataFrame,
    available_models: Sequence[str],
) -> str:
    """Apply the preregistered feature-family gate using development only.

    An augmented ridge may replace ``ridge_core`` only when all three strict
    conditions hold relative to core on the same OOS development predictions:

    * pooled MSE is lower (equivalently, pooled OOS R-squared improves because
      every model shares the same fold-historical-mean denominator);
    * MSE is lower in at least two assets; and
    * MSE is lower in a strict majority of asset/crop-year folds.

    If multiple candidates clear the gate, the one with lowest pooled MSE is
    selected; the frozen candidate order breaks exact ties.  No holdout row is
    ever consulted.
    """

    models = tuple(dict.fromkeys(map(str, available_models)))
    if RIDGE_CORE not in models:
        raise ValueError("ridge_core is required for development selection")
    candidates = tuple(model for model in RIDGE_CANDIDATE_ORDER if model in models)
    if not candidates:
        return RIDGE_CORE

    frame = development_predictions.copy()
    if "split" in frame:
        frame = frame.loc[frame["split"].eq("development")]
    frame = frame.loc[frame["model"].isin((RIDGE_CORE, *candidates))]
    if frame.empty or not frame["model"].eq(RIDGE_CORE).any():
        raise ValueError("development predictions contain no ridge_core rows")

    pooled = _grouped_model_mse(frame, ())
    by_asset = _grouped_model_mse(frame, ("asset",))
    by_fold = _grouped_model_mse(frame, ("asset", "crop_year"))
    core_pooled = float(pooled.get(RIDGE_CORE, np.nan))
    if not np.isfinite(core_pooled):
        raise ValueError("ridge_core development MSE is not finite")

    eligible: list[tuple[float, int, str]] = []
    for order, candidate in enumerate(candidates):
        candidate_pooled = float(pooled.get(candidate, np.nan))
        pooled_improves = bool(
            np.isfinite(candidate_pooled) and candidate_pooled < core_pooled
        )

        core_assets = by_asset.xs(RIDGE_CORE, level="model")
        try:
            candidate_assets = by_asset.xs(candidate, level="model")
        except KeyError:
            candidate_assets = pd.Series(dtype=float)
        common_assets = core_assets.index.intersection(candidate_assets.index)
        assets_improved = int(
            (candidate_assets.reindex(common_assets) < core_assets.reindex(common_assets)).sum()
        )

        core_folds = by_fold.xs(RIDGE_CORE, level="model")
        try:
            candidate_folds = by_fold.xs(candidate, level="model")
        except KeyError:
            candidate_folds = pd.Series(dtype=float)
        common_folds = core_folds.index.intersection(candidate_folds.index)
        folds_improved = int(
            (candidate_folds.reindex(common_folds) < core_folds.reindex(common_folds)).sum()
        )
        majority_folds_improve = bool(
            len(common_folds) > 0 and 2 * folds_improved > len(common_folds)
        )
        if pooled_improves and assets_improved >= 2 and majority_folds_improve:
            eligible.append((candidate_pooled, order, candidate))

    if not eligible:
        return RIDGE_CORE
    return min(eligible, key=lambda value: (value[0], value[1]))[2]


def development_selection_plan(
    weekly_panel: pd.DataFrame,
    folds: Sequence[CropYearFold],
    config: PredictiveExperimentConfig | None = None,
) -> tuple[pd.Timestamp, tuple[tuple[str, int], ...]]:
    """Return globally chronological folds eligible for family selection.

    Grain crop years begin in different months. A shared cutoff prevents a
    later wheat development fold from using outcomes concurrent with an
    already locked corn or soybean holdout. Only whole development folds whose
    final label exit precedes the earliest holdout start and embargo survive.
    """

    cfg = config or PredictiveExperimentConfig()
    holdout_folds = [fold for fold in folds if fold.is_holdout]
    if not holdout_folds:
        raise ValueError("a global selection plan requires holdout folds")
    earliest_holdout_start = min(
        min(pd.Timestamp(date) for date in fold.test_dates)
        for fold in holdout_folds
    )
    cutoff = pd.Timestamp(earliest_holdout_start) - BDay(cfg.embargo_business_days)
    eligible: list[tuple[str, int]] = []
    for fold in folds:
        if fold.is_holdout:
            continue
        frame = weekly_panel.xs(fold.asset, level="asset").reindex(
            pd.DatetimeIndex(fold.test_dates)
        )
        target_end = pd.to_datetime(frame["target_end_date"], errors="coerce")
        if target_end.notna().all() and target_end.max() < cutoff:
            eligible.append((str(fold.asset), int(fold.test_crop_year)))
    eligible_tuple = tuple(sorted(eligible))
    if len({asset for asset, _ in eligible_tuple}) < 2:
        raise ValueError(
            "fewer than two assets have globally eligible development-selection folds"
        )
    return pd.Timestamp(cutoff), eligible_tuple


def _fold_table(
    folds: Sequence[CropYearFold],
    *,
    selection_cutoff: pd.Timestamp,
    selection_eligible: Sequence[tuple[str, int]],
) -> pd.DataFrame:
    eligible = set(selection_eligible)
    return pd.DataFrame(
        [
            {
                "asset": fold.asset,
                "test_crop_year": fold.test_crop_year,
                "split": "holdout" if fold.is_holdout else "development",
                "train_crop_years": ",".join(map(str, fold.train_crop_years)),
                "train_start": min(fold.train_dates),
                "train_end": max(fold.train_dates),
                "test_start": min(fold.test_dates),
                "test_end": max(fold.test_dates),
                "purge_cutoff": fold.purge_cutoff,
                "train_observations": len(fold.train_dates),
                "test_observations": len(fold.test_dates),
                "global_selection_cutoff": pd.Timestamp(selection_cutoff),
                "selection_eligible": (
                    str(fold.asset), int(fold.test_crop_year)
                ) in eligible,
            }
            for fold in folds
        ]
    )


def run_predictive_experiment(
    market_prices: pd.DataFrame,
    usda_features: pd.DataFrame | None = None,
    weather_features: pd.DataFrame | None = None,
    config: PredictiveExperimentConfig | None = None,
    *,
    proxy_input: bool = False,
    holdout_model_override: str | None = None,
) -> PredictiveExperimentResult:
    """Run development model selection and the locked selected-family holdout."""

    cfg = config or PredictiveExperimentConfig()
    daily, feature_groups = build_daily_prediction_panel(
        market_prices,
        usda_features=usda_features,
        weather_features=weather_features,
        config=cfg,
        proxy_input=proxy_input,
    )
    weekly = sample_weekly_panel(daily, cfg)
    folds: list[CropYearFold] = []
    by_asset: dict[str, tuple[CropYearFold, ...]] = {}
    holdout_years: dict[str, tuple[int, ...]] = {}
    for asset in cfg.assets:
        asset_folds = make_crop_year_folds(weekly, asset, cfg)
        by_asset[asset] = asset_folds
        folds.extend(asset_folds)
        holdout_years[asset] = tuple(
            fold.test_crop_year for fold in asset_folds if fold.is_holdout
        )

    selection_cutoff, selection_eligible = development_selection_plan(
        weekly, folds, cfg
    )
    selection_eligible_set = set(selection_eligible)

    model_columns = _model_feature_columns(feature_groups)
    development_predictions: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    hyperparameter_records: list[dict[str, Any]] = []
    for asset, asset_folds in by_asset.items():
        frame = weekly.xs(asset, level="asset").sort_index()
        for fold in asset_folds:
            if fold.is_holdout or (
                str(fold.asset), int(fold.test_crop_year)
            ) not in selection_eligible_set:
                continue
            prediction, coefficient, hyperparameter = _run_fold_models(
                frame, fold, model_columns, cfg, "development"
            )
            development_predictions.extend(prediction)
            coefficient_records.extend(coefficient)
            hyperparameter_records.extend(hyperparameter)
    development_frame = pd.DataFrame(development_predictions)
    if development_frame.empty:
        raise ValueError("development folds produced no predictions")
    development_selected_model = _select_development_model(
        development_frame,
        tuple(model_columns),
    )
    if holdout_model_override is not None:
        if str(holdout_model_override) not in model_columns:
            raise ValueError(
                f"holdout model override is unavailable: {holdout_model_override}"
            )
        selected_model = str(holdout_model_override)
    else:
        selected_model = development_selected_model

    holdout_predictions: list[dict[str, Any]] = []
    holdout_model_names = tuple(dict.fromkeys((RIDGE_CORE, selected_model)))
    holdout_columns = {
        model: model_columns[model]
        for model in holdout_model_names
    }
    for asset, asset_folds in by_asset.items():
        frame = weekly.xs(asset, level="asset").sort_index()
        for fold in asset_folds:
            if not fold.is_holdout:
                continue
            prediction, coefficient, hyperparameter = _run_fold_models(
                frame, fold, holdout_columns, cfg, "holdout"
            )
            holdout_predictions.extend(prediction)
            coefficient_records.extend(coefficient)
            hyperparameter_records.extend(hyperparameter)

    predictions = pd.concat(
        [development_frame, pd.DataFrame(holdout_predictions)], ignore_index=True
    )
    predictions = predictions.sort_values(
        ["split", "date", "asset", "model"], kind="stable"
    ).reset_index(drop=True)
    metrics_overall = _metrics_table(predictions, ["split", "model"])
    metrics_by_asset = _metrics_table(predictions, ["split", "model", "asset"])
    metrics_by_crop_year = _metrics_table(
        predictions, ["split", "model", "asset", "crop_year"]
    )
    coefficients = pd.DataFrame(coefficient_records).sort_values(
        ["split", "asset", "crop_year", "model", "feature"], kind="stable"
    ).reset_index(drop=True)
    hyperparameters = pd.DataFrame(hyperparameter_records).sort_values(
        ["split", "asset", "crop_year", "model", "alpha"], kind="stable"
    ).reset_index(drop=True)
    return PredictiveExperimentResult(
        config=cfg,
        proxy_input=bool(proxy_input),
        selected_model=selected_model,
        feature_groups=feature_groups,
        holdout_crop_years=holdout_years,
        daily_panel=daily,
        weekly_panel=weekly,
        folds=_fold_table(
            folds,
            selection_cutoff=selection_cutoff,
            selection_eligible=selection_eligible,
        ),
        predictions=predictions,
        coefficients=coefficients,
        hyperparameters=hyperparameters,
        metrics_overall=metrics_overall,
        metrics_by_asset=metrics_by_asset,
        metrics_by_crop_year=metrics_by_crop_year,
        development_selected_model=development_selected_model,
    )


__all__ = [
    "ASSET_PRICE_ALIASES",
    "BASELINE_MODELS",
    "CORE_FEATURE_COLUMNS",
    "CROP_YEAR_START_MONTH",
    "PROXY_CONTRACT_MONTHS",
    "RIDGE_CANDIDATE_ORDER",
    "CropYearFold",
    "PredictiveExperimentConfig",
    "PredictiveExperimentResult",
    "asset_crop_year",
    "build_daily_prediction_panel",
    "development_selection_plan",
    "make_crop_year_folds",
    "run_predictive_experiment",
    "sample_weekly_panel",
]
