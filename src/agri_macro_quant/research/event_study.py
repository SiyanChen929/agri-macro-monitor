"""Event studies with regime-aware CAR paths and market-model adjustments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    """Detailed event paths, terminal event CARs, and grouped statistics."""

    paths: pd.DataFrame
    event_cars: pd.DataFrame
    summary: pd.DataFrame

    def __getitem__(self, key: str) -> pd.DataFrame:
        return getattr(self, key)

    def to_dict(self) -> dict[str, pd.DataFrame]:
        return {"paths": self.paths, "event_cars": self.event_cars, "summary": self.summary}


def _returns_frame(returns: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        frame = returns.to_frame(returns.name or "asset")
    elif isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")
    if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index)
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index().apply(pd.to_numeric, errors="coerce")


def _events_frame(events: pd.DataFrame | pd.Series | pd.Index | list[Any]) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        frame = events.copy()
        if "date" not in frame.columns:
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    elif isinstance(events, pd.Series):
        if isinstance(events.index, pd.DatetimeIndex):
            frame = pd.DataFrame({"date": events.index, "event_type": events.to_numpy()})
        else:
            frame = pd.DataFrame({"date": events.to_numpy()})
    else:
        frame = pd.DataFrame({"date": list(events)})
    if "date" not in frame.columns:
        raise ValueError("events require a date column or datetime index")
    frame["date"] = pd.to_datetime(frame["date"])
    if "event_type" not in frame.columns:
        frame["event_type"] = "event"
    if "event_id" not in frame.columns:
        frame["event_id"] = np.arange(len(frame), dtype=int)
    return frame.sort_values("date", ignore_index=True)


def _benchmark_series(
    benchmark_returns: pd.Series | pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series | None:
    if benchmark_returns is None:
        return None
    if isinstance(benchmark_returns, pd.DataFrame):
        if benchmark_returns.shape[1] != 1:
            raise ValueError("benchmark_returns DataFrame must have exactly one column")
        series = benchmark_returns.iloc[:, 0]
    else:
        series = benchmark_returns.copy()
    series.index = pd.to_datetime(series.index)
    return pd.to_numeric(series, errors="coerce").reindex(index)


def _aligned_event_position(index: pd.DatetimeIndex, event_date: pd.Timestamp, alignment: str) -> int | None:
    side = "left" if alignment in {"next", "on_or_after"} else "right"
    position = int(index.searchsorted(event_date, side=side))
    if side == "right":
        position -= 1
    if position < 0 or position >= len(index):
        return None
    return position


def _market_model(
    asset: pd.Series,
    benchmark: pd.Series | None,
    event_position: int,
    estimation_window: int,
    estimation_gap: int,
    min_estimation_obs: int,
) -> tuple[float, float]:
    if benchmark is None:
        return 0.0, 0.0
    end = max(0, event_position - estimation_gap)
    start = max(0, end - estimation_window)
    sample = pd.concat(
        [asset.iloc[start:end].rename("asset"), benchmark.iloc[start:end].rename("benchmark")],
        axis=1,
    ).dropna()
    if len(sample) < min_estimation_obs or float(sample["benchmark"].var(ddof=0)) <= 0:
        return 0.0, 1.0
    beta = float(sample["asset"].cov(sample["benchmark"]) / sample["benchmark"].var(ddof=1))
    alpha = float(sample["asset"].mean() - beta * sample["benchmark"].mean())
    return alpha, beta


def _two_sided_pvalue(t_stat: float, degrees_freedom: int) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(2.0 * student_t.sf(abs(t_stat), max(1, degrees_freedom)))
    except ImportError:
        # Normal approximation is accurate enough for larger event samples and
        # keeps scipy optional for the MVP.
        return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _summarize_event_cars(cars: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cars.empty:
        return pd.DataFrame(
            columns=group_columns
            + ["n_events", "mean_car", "median_car", "std_car", "standard_error", "t_stat", "p_value", "hit_rate"]
        )
    grouped: Any = cars.groupby(group_columns, dropna=False) if group_columns else [((), cars)]
    for key, group in grouped:
        values = pd.to_numeric(group["car"], errors="coerce").dropna()
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        n = int(len(values))
        mean = float(values.mean()) if n else np.nan
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        standard_error = std / math.sqrt(n) if n > 1 else np.nan
        t_stat = mean / standard_error if standard_error and standard_error > 0 else np.nan
        row.update(
            {
                "n_events": n,
                "mean_car": mean,
                "median_car": float(values.median()) if n else np.nan,
                "std_car": std,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "p_value": _two_sided_pvalue(t_stat, n - 1),
                "hit_rate": float((values > 0).mean()) if n else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_event_study(
    returns: pd.DataFrame | pd.Series,
    events: pd.DataFrame | pd.Series | pd.Index | list[Any],
    pre_window: int = 5,
    post_window: int = 5,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> EventStudyResult:
    """Calculate abnormal-return and CAR paths around dated events.

    Event rows may include ``event_type``, ``regime``, ``commodity``, and any
    other metadata; these fields are retained.  By default, non-trading-day
    events align to the next trading day.  A supplied benchmark invokes a
    rolling pre-event market model; otherwise raw returns are abnormal returns.
    """

    cfg = dict(config or {})
    frame = _returns_frame(returns)
    event_frame = _events_frame(events)
    benchmark = _benchmark_series(benchmark_returns, frame.index)
    pre = max(0, int(pre_window))
    post = max(0, int(post_window))
    estimation_window = max(2, int(cfg.get("estimation_window", 120)))
    estimation_gap = max(0, int(cfg.get("estimation_gap", pre + 1)))
    min_estimation_obs = max(2, int(cfg.get("min_estimation_obs", min(30, estimation_window))))
    alignment = str(cfg.get("event_date_alignment", "next")).lower()
    require_full = bool(cfg.get("require_full_window", True))
    compound_car = bool(cfg.get("compound_car", False))
    selected_assets = cfg.get("assets", list(frame.columns))

    path_rows: list[dict[str, Any]] = []
    reserved = {
        "event_date", "aligned_event_date", "asset", "relative_day", "observation_date",
        "return", "expected_return", "abnormal_return", "car", "alpha", "beta",
    }
    metadata_source = [column for column in event_frame.columns if column != "date"]
    metadata_names = {
        column: (f"event_{column}" if column in reserved else column)
        for column in metadata_source
    }
    metadata_columns = list(metadata_names.values())
    for _, event in event_frame.iterrows():
        event_date = pd.Timestamp(event["date"])
        event_position = _aligned_event_position(frame.index, event_date, alignment)
        if event_position is None:
            continue
        start = event_position - pre
        end = event_position + post
        if require_full and (start < 0 or end >= len(frame)):
            continue
        start = max(0, start)
        end = min(len(frame) - 1, end)
        actual_event_date = frame.index[event_position]
        event_assets = selected_assets
        if "asset" in event.index and pd.notna(event["asset"]):
            event_assets = event["asset"] if isinstance(event["asset"], (list, tuple, set)) else [event["asset"]]
        for asset in event_assets:
            if asset not in frame.columns:
                continue
            alpha, beta = _market_model(
                frame[asset], benchmark, event_position, estimation_window, estimation_gap, min_estimation_obs
            )
            car = 0.0
            for position in range(start, end + 1):
                observed = frame.iloc[position][asset]
                benchmark_return = benchmark.iloc[position] if benchmark is not None else 0.0
                expected = alpha + beta * benchmark_return if pd.notna(benchmark_return) else np.nan
                abnormal = observed - expected if pd.notna(observed) and pd.notna(expected) else np.nan
                if pd.notna(abnormal):
                    car = (1.0 + car) * (1.0 + abnormal) - 1.0 if compound_car else car + abnormal
                row: dict[str, Any] = {
                    "event_date": event_date,
                    "aligned_event_date": actual_event_date,
                    "asset": str(asset),
                    "relative_day": position - event_position,
                    "observation_date": frame.index[position],
                    "return": observed,
                    "expected_return": expected,
                    "abnormal_return": abnormal,
                    "car": car,
                    "alpha": alpha,
                    "beta": beta,
                }
                for source_column, output_column in metadata_names.items():
                    row[output_column] = event[source_column]
                path_rows.append(row)

    paths = pd.DataFrame(path_rows)
    base_columns = [
        "event_date", "aligned_event_date", "asset", "relative_day", "observation_date",
        "return", "expected_return", "abnormal_return", "car", "alpha", "beta",
    ]
    if paths.empty:
        paths = pd.DataFrame(columns=base_columns + metadata_columns)
        cars = pd.DataFrame(columns=["event_date", "asset", "car"] + metadata_columns)
        summary = _summarize_event_cars(cars, ["event_type", "asset"])
        return EventStudyResult(paths=paths, event_cars=cars, summary=summary)

    terminal = paths.sort_values("relative_day").groupby(["event_id", "asset"], as_index=False).tail(1)
    car_columns = ["event_date", "aligned_event_date", "asset", "car", "alpha", "beta"]
    car_columns += [column for column in metadata_columns if column not in car_columns]
    cars = terminal[car_columns].reset_index(drop=True)
    group_columns = list(cfg.get("group_by", ["event_type", "asset"]))
    if bool(cfg.get("condition_on_regime", True)) and "regime" in cars.columns and "regime" not in group_columns:
        group_columns.append("regime")
    group_columns = [column for column in group_columns if column in cars.columns]
    summary = _summarize_event_cars(cars, group_columns)
    return EventStudyResult(paths=paths, event_cars=cars, summary=summary)


event_study = run_event_study


__all__ = ["EventStudyResult", "event_study", "run_event_study"]
