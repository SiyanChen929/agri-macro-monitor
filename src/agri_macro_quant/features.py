"""Point-in-time-safe feature engineering utilities.

All rolling operations in this module are trailing (never centred).  By default
the observation at ``t`` is allowed in the statistic at ``t``; set
``history_lag=1`` when a strategy must form a signal before observing ``x_t``.

The public functions accept ordinary pandas objects and plain dictionaries so
they can be used independently of the ingestion/configuration layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1.0e-12


def _as_numeric_series(values: pd.Series, *, name: str | None = None) -> pd.Series:
    """Return a float Series while preserving the input index."""

    if not isinstance(values, pd.Series):
        values = pd.Series(values, name=name)
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if name is not None:
        result.name = name
    return result


def _minimum_periods(window: int, min_periods: int | None) -> int:
    if window < 1:
        raise ValueError("window must be at least 1")
    if min_periods is None:
        return min(window, max(2, window // 4))
    if not 1 <= int(min_periods) <= window:
        raise ValueError("min_periods must be between 1 and window")
    return int(min_periods)


def _history(values: pd.Series, history_lag: int) -> pd.Series:
    if history_lag < 0:
        raise ValueError("history_lag cannot be negative")
    return values.shift(int(history_lag)) if history_lag else values


def rolling_winsorize(
    values: pd.Series,
    window: int = 252,
    limits: tuple[float, float] = (0.01, 0.99),
    *,
    min_periods: int | None = None,
    history_lag: int = 0,
) -> pd.Series:
    """Causally clip observations to trailing empirical quantiles.

    Quantile bounds use data no later than ``t-history_lag``.  With
    ``history_lag=1`` the current observation is clipped using only prior data.
    """

    x = _as_numeric_series(values)
    lower, upper = (float(limits[0]), float(limits[1]))
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("limits must satisfy 0 <= lower < upper <= 1")
    periods = _minimum_periods(int(window), min_periods)
    history = _history(x, history_lag)
    roll = history.rolling(int(window), min_periods=periods)
    lo = roll.quantile(lower)
    hi = roll.quantile(upper)
    result = x.clip(lower=lo, upper=hi)
    result.name = values.name
    return result


def rolling_zscore(
    values: pd.Series,
    window: int = 252,
    *,
    min_periods: int | None = None,
    ddof: int = 0,
    history_lag: int = 0,
    clip: float | None = None,
) -> pd.Series:
    """Trailing standard z-score with safe handling of constant histories."""

    x = _as_numeric_series(values)
    periods = _minimum_periods(int(window), min_periods)
    history = _history(x, history_lag)
    roll = history.rolling(int(window), min_periods=periods)
    centre = roll.mean()
    scale = roll.std(ddof=int(ddof))
    result = (x - centre) / scale.mask(scale.abs() <= EPSILON)
    # A value equal to a constant, sufficiently long history has zero surprise.
    constant = scale.notna() & (scale.abs() <= EPSILON) & x.notna() & centre.notna()
    result = result.mask(constant & ((x - centre).abs() <= EPSILON), 0.0)
    if clip is not None:
        jump = constant & ((x - centre).abs() > EPSILON)
        result = result.mask(jump, np.sign(x - centre) * abs(float(clip)))
    if clip is not None:
        result = result.clip(-abs(float(clip)), abs(float(clip)))
    result.name = values.name
    return result


def robust_zscore(
    values: pd.Series,
    window: int = 252,
    *,
    min_periods: int | None = None,
    history_lag: int = 0,
    winsor_limits: tuple[float, float] | None = (0.01, 0.99),
    clip: float | None = 5.0,
) -> pd.Series:
    """Trailing robust z-score based on the median and scaled MAD.

    If MAD is zero, the trailing IQR is used; if both are zero, an observation
    equal to the rolling median receives a score of zero.
    """

    x = _as_numeric_series(values)
    periods = _minimum_periods(int(window), min_periods)
    if winsor_limits is not None:
        x_for_score = rolling_winsorize(
            x,
            int(window),
            winsor_limits,
            min_periods=periods,
            history_lag=history_lag,
        )
    else:
        x_for_score = x
    history = _history(x, history_lag)
    roll = history.rolling(int(window), min_periods=periods)
    centre = roll.median()
    # Compute each window around its own median.  ``nanmedian`` is required
    # because pandas may pass the full window (including missing entries) once
    # the non-missing count reaches ``min_periods``.
    mad = roll.apply(
        lambda a: float(np.nanmedian(np.abs(a - np.nanmedian(a)))), raw=True
    )
    scale = 1.4826 * mad
    iqr_scale = (roll.quantile(0.75) - roll.quantile(0.25)) / 1.349
    scale = scale.where(scale.abs() > EPSILON, iqr_scale)
    result = (x_for_score - centre) / scale.mask(scale.abs() <= EPSILON)
    constant = scale.notna() & (scale.abs() <= EPSILON) & x_for_score.notna()
    result = result.mask(constant & ((x_for_score - centre).abs() <= EPSILON), 0.0)
    if clip is not None:
        jump = constant & ((x_for_score - centre).abs() > EPSILON)
        result = result.mask(
            jump, np.sign(x_for_score - centre) * abs(float(clip))
        )
    if clip is not None:
        result = result.clip(-abs(float(clip)), abs(float(clip)))
    result.name = values.name
    return result


def expanding_zscore(
    values: pd.Series,
    *,
    min_periods: int = 20,
    robust: bool = False,
    history_lag: int = 0,
    clip: float | None = None,
) -> pd.Series:
    """Expanding, point-in-time-safe z-score."""

    x = _as_numeric_series(values)
    history = _history(x, history_lag)
    expanding = history.expanding(min_periods=max(2, int(min_periods)))
    if robust:
        centre = expanding.median()
        mad = expanding.apply(
            lambda a: float(np.nanmedian(np.abs(a - np.nanmedian(a)))), raw=True
        )
        scale = 1.4826 * mad
    else:
        centre = expanding.mean()
        scale = expanding.std(ddof=0)
    result = (x - centre) / scale.mask(scale.abs() <= EPSILON)
    constant = scale.notna() & (scale.abs() <= EPSILON) & x.notna()
    result = result.mask(constant & ((x - centre).abs() <= EPSILON), 0.0)
    if clip is not None:
        jump = constant & ((x - centre).abs() > EPSILON)
        result = result.mask(jump, np.sign(x - centre) * abs(float(clip)))
    if clip is not None:
        result = result.clip(-abs(float(clip)), abs(float(clip)))
    result.name = values.name
    return result


def rolling_percentile_rank(
    values: pd.Series,
    window: int = 252,
    *,
    min_periods: int | None = None,
    history_lag: int = 0,
) -> pd.Series:
    """Trailing percentile rank in ``[0, 1]`` using a mid-rank for ties."""

    x = _as_numeric_series(values)
    periods = _minimum_periods(int(window), min_periods)
    if history_lag:
        # Rank x_t against its prior history, including x_t only as the value
        # being located rather than as an input to the empirical distribution.
        hist = _history(x, history_lag)
        # This explicit loop gives current-vs-history semantics for any lag.
        out = pd.Series(np.nan, index=x.index, dtype=float)
        array = hist.to_numpy(dtype=float)
        current = x.to_numpy(dtype=float)
        for position in range(len(x)):
            start = max(0, position - int(window) + 1)
            sample = array[start : position + 1]
            sample = sample[np.isfinite(sample)]
            if sample.size >= periods and np.isfinite(current[position]):
                out.iloc[position] = (
                    np.sum(sample < current[position])
                    + 0.5 * np.sum(sample == current[position])
                ) / sample.size
        out.name = values.name
        return out

    def _last_rank(array: np.ndarray) -> float:
        last = array[-1]
        if not np.isfinite(last):
            return np.nan
        sample = array[np.isfinite(array)]
        return float(
            (np.sum(sample < last) + 0.5 * np.sum(sample == last)) / len(sample)
        )

    result = x.rolling(int(window), min_periods=periods).apply(_last_rank, raw=True)
    result.name = values.name
    return result


def rolling_regression_slope(
    values: pd.Series,
    window: int = 20,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """OLS slope against a local integer time trend over a trailing window."""

    x = _as_numeric_series(values)
    periods = _minimum_periods(int(window), min_periods)

    def _slope(array: np.ndarray) -> float:
        mask = np.isfinite(array)
        if mask.sum() < 2:
            return np.nan
        y = array[mask]
        t = np.arange(array.size, dtype=float)[mask]
        t = t - t.mean()
        denominator = float(t @ t)
        return float((t @ (y - y.mean())) / denominator) if denominator else 0.0

    result = x.rolling(int(window), min_periods=periods).apply(_slope, raw=True)
    result.name = values.name
    return result


def last_known_with_age(
    values: pd.Series,
    *,
    max_age_days: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Carry the last release forward and return its age in calendar days.

    This is intended for weekly/monthly releases.  Values older than
    ``max_age_days`` are invalidated rather than carried indefinitely.
    """

    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("last_known_with_age requires a DatetimeIndex")
    x = _as_numeric_series(values).sort_index()
    release_dates = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")
    release_dates.loc[x.notna()] = x.index[x.notna()]
    release_dates = release_dates.ffill()
    index_series = pd.Series(x.index, index=x.index)
    age = (index_series - release_dates).dt.total_seconds() / 86_400.0
    filled = x.ffill()
    if max_age_days is not None:
        stale = age > int(max_age_days)
        filled = filled.mask(stale)
        age = age.mask(stale)
    filled.name = values.name
    age.name = f"{values.name}__data_age_days"
    return filled, age.astype(float)


def apply_release_lags(
    frame: pd.DataFrame,
    lags: Mapping[str, int] | None,
) -> pd.DataFrame:
    """Shift columns by row periods to approximate their publication lag."""

    result = frame.copy()
    for column, lag in (lags or {}).items():
        if column in result:
            if int(lag) < 0:
                raise ValueError(f"publication lag for {column!r} cannot be negative")
            result[column] = result[column].shift(int(lag))
    return result


def point_in_time_asof(
    records: pd.DataFrame,
    calendar: pd.DatetimeIndex | Sequence[pd.Timestamp] | None = None,
    *,
    series_col: str = "series",
    value_col: str = "value",
    observation_col: str = "observation_date",
    publication_col: str = "publication_date",
    revision_col: str = "revision_date",
    effective_col: str = "effective_date",
    max_age_days: int | Mapping[str, int] | None = None,
    decision_time: str | pd.Timedelta | None = "23:59:59",
) -> pd.DataFrame:
    """Materialise last-known values from a long release/vintage table.

    For every as-of date and series, this selects the latest observation that
    had actually been released, then the most recent known revision of that
    observation.  A revision of an older observation cannot displace a newer
    observation.  ``effective_date`` is honoured when present; otherwise the
    later of publication and revision dates is the information timestamp.
    Date-only calendars are evaluated at end of day by default; pass
    ``decision_time=None`` to treat calendar timestamps literally.
    """

    required = {series_col, value_col, observation_col, publication_col}
    missing = required.difference(records.columns)
    if missing:
        raise KeyError(f"point-in-time records missing columns: {sorted(missing)}")
    data = records.copy()
    for column in (observation_col, publication_col, revision_col, effective_col):
        if column in data:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    # Never expose a value before publication.  When effective/revision dates
    # exist, the value becomes known at the latest applicable timestamp.
    information_columns = [publication_col]
    if revision_col in data:
        information_columns.append(revision_col)
    if effective_col in data:
        information_columns.append(effective_col)
    information_date = data[information_columns].max(axis=1)
    data = data.assign(__information_date=information_date)
    data = data.dropna(subset=[series_col, observation_col, "__information_date"])
    if calendar is None:
        if data.empty:
            return pd.DataFrame()
        calendar_index = pd.date_range(
            data["__information_date"].min().normalize(),
            data["__information_date"].max().normalize(),
            freq="D",
        )
    else:
        calendar_index = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    decision_offset: pd.Timedelta | None
    if decision_time is None:
        decision_offset = None
    elif isinstance(decision_time, pd.Timedelta):
        decision_offset = decision_time
    else:
        parts = [int(part) for part in str(decision_time).split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError("decision_time must use HH:MM or HH:MM:SS")
        hours, minutes = parts[0], parts[1]
        seconds = parts[2] if len(parts) == 3 else 0
        if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
            raise ValueError("decision_time is outside the valid clock range")
        decision_offset = pd.Timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
    output = pd.DataFrame(index=calendar_index)
    for series_name, group in data.groupby(series_col, sort=False):
        group = group.sort_values(["__information_date", observation_col])
        values: list[Any] = []
        ages: list[float] = []
        known: dict[pd.Timestamp, tuple[pd.Timestamp, Any]] = {}
        rows = list(group.itertuples(index=False, name=None))
        columns = list(group.columns)
        info_pos = columns.index("__information_date")
        obs_pos = columns.index(observation_col)
        value_pos = columns.index(value_col)
        cursor = 0
        for asof in calendar_index:
            decision_at = (
                asof if decision_offset is None else asof.normalize() + decision_offset
            )
            while cursor < len(rows) and rows[cursor][info_pos] <= decision_at:
                row = rows[cursor]
                obs = row[obs_pos]
                info = row[info_pos]
                previous = known.get(obs)
                if previous is None or info >= previous[0]:
                    known[obs] = (info, row[value_pos])
                cursor += 1
            eligible_observations = [obs for obs in known if obs <= decision_at]
            if not eligible_observations:
                values.append(np.nan)
                ages.append(np.nan)
                continue
            latest_observation = max(eligible_observations)
            values.append(known[latest_observation][1])
            ages.append(float((asof - latest_observation).total_seconds() / 86_400.0))
        series = pd.to_numeric(pd.Series(values, index=calendar_index), errors="coerce")
        age_series = pd.Series(ages, index=calendar_index, dtype=float)
        threshold = (
            max_age_days.get(str(series_name))
            if isinstance(max_age_days, Mapping)
            else max_age_days
        )
        if threshold is not None:
            stale = age_series > int(threshold)
            series = series.mask(stale)
            age_series = age_series.mask(stale)
        output[str(series_name)] = series
        output[f"{series_name}__data_age_days"] = age_series
    output.index.name = "date"
    return output


DEFAULT_FEATURE_CONFIG: dict[str, Any] = {
    "include_original": True,
    "changes": [5, 20, 60, 120],
    "zscore_windows": [60, 252],
    "percentile_windows": [252],
    "momentum_windows": [20, 60],
    "volatility_windows": [20],
    "trend_windows": [20, 60],
    "min_periods": None,
    "robust": True,
    "winsor_limits": (0.01, 0.99),
    "z_clip": 5.0,
    "history_lag": 0,
    "publication_lags": {},
    "carry_forward": {},
    "surprises": {},
}


def _merged_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_FEATURE_CONFIG)
    if config:
        supplied = dict(config)
        merged.update(supplied)
        if "rolling_windows" in supplied and "changes" not in supplied:
            merged["changes"] = supplied["rolling_windows"]
        if "winsor_quantiles" in supplied and "winsor_limits" not in supplied:
            merged["winsor_limits"] = tuple(supplied["winsor_quantiles"])
    return merged


def _configured_min_periods(cfg: Mapping[str, Any], window: int) -> int | None:
    value = cfg.get("min_periods")
    if isinstance(value, Mapping):
        value = value.get(window, value.get(str(window)))
    if value is not None:
        return int(value)
    fraction = cfg.get("minimum_periods_fraction")
    if fraction is not None:
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("minimum_periods_fraction must lie in (0, 1]")
        return min(window, max(2, int(np.ceil(window * fraction))))
    return None


def build_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a deterministic wide feature frame from wide, date-indexed data.

    Generated columns use ``<source>__<transform>`` names.  Important config
    keys are listed in :data:`DEFAULT_FEATURE_CONFIG`.  ``carry_forward`` maps
    a release column to its maximum permissible age in calendar days, and
    ``surprises`` maps a name to ``{"actual": ..., "expected": ...}``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    cfg = _merged_config(config)
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="raise")
    data = data.sort_index()
    if data.index.has_duplicates:
        data = data.loc[~data.index.duplicated(keep="last")]
    data = apply_release_lags(data, cfg.get("publication_lags"))

    age_features: dict[str, pd.Series] = {}
    for column, max_age in dict(cfg.get("carry_forward") or {}).items():
        if column in data:
            data[column], age = last_known_with_age(data[column], max_age_days=max_age)
            age_features[age.name] = age

    selected = cfg.get("columns")
    if selected is None:
        numeric_columns = list(data.select_dtypes(include=[np.number, "bool"]).columns)
    else:
        numeric_columns = [column for column in selected if column in data]

    output: dict[str, pd.Series] = {}
    if bool(cfg.get("include_original", True)):
        for column in data.columns:
            output[str(column)] = data[column]
    output.update(age_features)

    changes = [int(item) for item in cfg.get("changes", [])]
    z_windows = [int(item) for item in cfg.get("zscore_windows", [])]
    percentile_windows = [int(item) for item in cfg.get("percentile_windows", [])]
    momentum_windows = [int(item) for item in cfg.get("momentum_windows", [])]
    volatility_windows = [int(item) for item in cfg.get("volatility_windows", [])]
    trend_windows = [int(item) for item in cfg.get("trend_windows", [])]
    history_lag = int(cfg.get("history_lag", 0))

    for column in numeric_columns:
        name = str(column)
        x = _as_numeric_series(data[column], name=name)
        output[f"{name}__level"] = x
        previous = x.shift(1)
        log_return = np.log(x.where(x > 0) / previous.where(previous > 0))
        simple_return = x / previous.mask(previous.abs() <= EPSILON) - 1.0
        one_return = log_return.where((x > 0) & (previous > 0), simple_return)
        output[f"{name}__return_1"] = one_return.replace([np.inf, -np.inf], np.nan)
        for window in changes:
            output[f"{name}__change_{window}"] = x - x.shift(window)
        for window in z_windows:
            periods = _configured_min_periods(cfg, window)
            if cfg.get("robust", True):
                z = robust_zscore(
                    x,
                    window,
                    min_periods=periods,
                    history_lag=history_lag,
                    winsor_limits=cfg.get("winsor_limits"),
                    clip=cfg.get("z_clip"),
                )
            else:
                z = rolling_zscore(
                    x,
                    window,
                    min_periods=periods,
                    history_lag=history_lag,
                    clip=cfg.get("z_clip"),
                )
            output[f"{name}__z_{window}"] = z
        for window in percentile_windows:
            output[f"{name}__pct_rank_{window}"] = rolling_percentile_rank(
                x,
                window,
                min_periods=_configured_min_periods(cfg, window),
                history_lag=history_lag,
            )
        for window in momentum_windows:
            previous = x.shift(window)
            log_momentum = np.log(x.where(x > 0) / previous.where(previous > 0))
            arithmetic_momentum = x - previous
            momentum = log_momentum.where(
                (x > 0) & (previous > 0), arithmetic_momentum
            )
            output[f"{name}__momentum_{window}"] = momentum.replace(
                [np.inf, -np.inf], np.nan
            )
        for window in volatility_windows:
            output[f"{name}__rv_{window}"] = one_return.rolling(
                window,
                min_periods=_minimum_periods(
                    window, _configured_min_periods(cfg, window)
                ),
            ).std(ddof=0) * np.sqrt(252.0)
        for window in trend_windows:
            mean = x.rolling(
                window,
                min_periods=_minimum_periods(
                    window, _configured_min_periods(cfg, window)
                ),
            ).mean()
            output[f"{name}__ma_distance_{window}"] = (x / mean - 1.0).replace(
                [np.inf, -np.inf], np.nan
            )
            output[f"{name}__slope_{window}"] = rolling_regression_slope(
                x,
                window,
                min_periods=_configured_min_periods(cfg, window),
            )

    for surprise_name, definition in dict(cfg.get("surprises") or {}).items():
        if isinstance(definition, Mapping):
            actual = definition.get("actual")
            expected = definition.get("expected")
        else:
            actual, expected = definition
        if actual in data and expected in data:
            output[str(surprise_name)] = _as_numeric_series(data[actual]) - _as_numeric_series(
                data[expected]
            )

    result = pd.DataFrame(output, index=data.index)
    result.index.name = frame.index.name or "date"
    result.attrs["point_in_time_safe"] = True
    result.attrs["feature_config"] = cfg
    return result


# Friendly aliases used by notebooks and callers with older naming conventions.
make_features = build_features
winsorize_rolling = rolling_winsorize
rolling_robust_zscore = robust_zscore


__all__ = [
    "DEFAULT_FEATURE_CONFIG",
    "apply_release_lags",
    "build_features",
    "expanding_zscore",
    "last_known_with_age",
    "make_features",
    "point_in_time_asof",
    "robust_zscore",
    "rolling_percentile_rank",
    "rolling_regression_slope",
    "rolling_robust_zscore",
    "rolling_winsorize",
    "rolling_zscore",
    "winsorize_rolling",
]
