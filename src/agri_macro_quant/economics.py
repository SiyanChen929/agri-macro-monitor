"""Economically motivated physical-supply and input-margin calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data_models import PolicyEvent
from .features import robust_zscore


def calculate_exportable_surplus(
    country_fundamentals: pd.DataFrame,
    *,
    exporter_column: str = "is_major_exporter",
) -> pd.DataFrame:
    """Calculate exporter surplus and exporter stocks/use by date and commodity.

    Required columns are ``date``, ``country``, ``commodity``, ``production``,
    ``beginning_stocks``, ``imports``, ``domestic_consumption``,
    ``required_ending_stocks``, ``ending_stocks``, and ``exports``.  World
    producers not marked as major exporters are deliberately excluded.
    """

    required = {
        "date",
        "country",
        "commodity",
        "production",
        "beginning_stocks",
        "imports",
        "domestic_consumption",
        "required_ending_stocks",
        "ending_stocks",
        "exports",
        exporter_column,
    }
    missing = required - set(country_fundamentals.columns)
    if missing:
        raise ValueError(f"country fundamentals missing columns: {sorted(missing)}")
    frame = country_fundamentals.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    numeric = [
        "production",
        "beginning_stocks",
        "imports",
        "domestic_consumption",
        "required_ending_stocks",
        "ending_stocks",
        "exports",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    exporters = frame[frame[exporter_column].astype(bool)].copy()
    exporters["exportable_surplus"] = (
        exporters["production"]
        + exporters["beginning_stocks"]
        + exporters["imports"]
        - exporters["domestic_consumption"]
        - exporters["required_ending_stocks"]
    )
    grouped = exporters.groupby(["date", "commodity"], sort=True)
    result = grouped.agg(
        global_exportable_surplus=("exportable_surplus", "sum"),
        exporter_ending_stocks=("ending_stocks", "sum"),
        exporter_domestic_use=("domestic_consumption", "sum"),
        exporter_exports=("exports", "sum"),
        exporter_count=("country", "nunique"),
    )
    denominator = result["exporter_domestic_use"] + result["exporter_exports"]
    result["exporter_stocks_to_use"] = 100.0 * result["exporter_ending_stocks"] / denominator.replace(0, np.nan)
    return result.reset_index()


def policy_stress_series(
    events: Sequence[PolicyEvent] | pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    half_life_days: float = 30.0,
) -> pd.Series:
    """Aggregate signed, trade-share-weighted policy events with causal decay."""

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if isinstance(events, pd.DataFrame):
        required = {
            "date",
            "direction",
            "magnitude",
            "confidence",
            "trade_share",
            "importance",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"policy events missing columns: {sorted(missing)}")
        event_rows = events.copy()
        event_rows["date"] = pd.to_datetime(event_rows["date"], errors="raise")
    else:
        event_rows = pd.DataFrame(
            [
                {
                    "date": event.date,
                    "direction": event.direction,
                    "magnitude": event.magnitude,
                    "confidence": event.confidence,
                    "trade_share": event.trade_share,
                    "importance": event.importance,
                    "duration_days": event.duration_days,
                }
                for event in events
            ]
        )
    index = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    score = pd.Series(0.0, index=index, name="policy_stress")
    for row in event_rows.itertuples(index=False):
        event_date = pd.Timestamp(row.date).normalize()
        eligible = index >= event_date
        ages = (index[eligible] - event_date).days.astype(float)
        duration = float(getattr(row, "duration_days", np.inf))
        active = ages <= duration
        decay = np.exp(-np.log(2.0) * ages / float(half_life_days))
        contribution = (
            float(row.direction)
            * float(row.magnitude)
            * float(row.confidence)
            * float(row.trade_share)
            * float(row.importance)
            * decay
        )
        target = index[eligible]
        score.loc[target[active]] += contribution[active]
    return score


def rolling_input_beta(
    product_price: pd.Series,
    feedstock_price: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
    fallback: float = 2.0,
    lower: float = 0.0,
    upper: float = 100.0,
) -> pd.Series:
    """Causal rolling beta of product changes to feedstock changes."""

    product = pd.to_numeric(product_price, errors="coerce").astype(float)
    feedstock = pd.to_numeric(feedstock_price, errors="coerce").astype(float)
    x = feedstock.diff()
    y = product.diff()
    covariance = x.rolling(window, min_periods=min_periods).cov(y)
    variance = x.rolling(window, min_periods=min_periods).var(ddof=0)
    beta = covariance / variance.replace(0, np.nan)
    return beta.shift(1).clip(lower, upper).fillna(float(fallback)).rename("input_beta")


def build_economic_features(
    market: pd.DataFrame,
    *,
    normalization_window: int = 252,
    min_periods: int = 20,
    nitrogen_beta: float | str = "empirical",
) -> pd.DataFrame:
    """Build margin, affordability, and cross-market features for signals.

    Only trailing values are used.  Missing source columns result in missing
    derived columns rather than guessed data.
    """

    data = market.copy().sort_index().apply(pd.to_numeric, errors="coerce")
    output = pd.DataFrame(index=data.index)

    def z(values: pd.Series) -> pd.Series:
        return robust_zscore(
            values,
            window=normalization_window,
            min_periods=min_periods,
            history_lag=0,
            clip=4.0,
        )

    ratios: list[pd.Series] = []
    if {"corn", "urea"}.issubset(data):
        raw = data["corn"] / data["urea"].replace(0, np.nan)
        output["corn_urea_raw"] = raw
        output["corn_urea"] = z(raw)
        ratios.append(output["corn_urea"])
    if {"wheat", "urea"}.issubset(data):
        raw = data["wheat"] / data["urea"].replace(0, np.nan)
        output["wheat_urea_raw"] = raw
        output["wheat_urea"] = z(raw)
        ratios.append(output["wheat_urea"])
    if {"soybeans", "urea"}.issubset(data):
        raw = data["soybeans"] / data["urea"].replace(0, np.nan)
        output["soy_fertilizer_raw"] = raw
        output["soy_fertilizer"] = z(raw)
        ratios.append(output["soy_fertilizer"])
    if ratios:
        output["ag_affordability"] = pd.concat(ratios, axis=1).mean(axis=1)
        output["farmer_margin"] = output["ag_affordability"]

    if {"urea", "natural_gas"}.issubset(data):
        if str(nitrogen_beta).lower() == "empirical":
            beta = rolling_input_beta(data["urea"], data["natural_gas"])
        else:
            beta = pd.Series(float(nitrogen_beta), index=data.index, name="input_beta")
        margin = data["urea"] - beta * data["natural_gas"]
        output["nitrogen_input_beta"] = beta
        output["nitrogen_margin_raw"] = margin
        output["nitrogen_margin"] = z(margin)

    momentum_names = {
        "wheat": "wheat_momentum",
        "corn": "corn_momentum",
        "soybeans": "soybean_momentum",
        "cf": "cf_momentum",
        "ntr": "ntr_momentum",
        "mos": "mos_momentum",
        "dxy": "dxy_momentum",
        "us_2y": "us2y_momentum",
        "natural_gas": "natural_gas",
    }
    for source, target in momentum_names.items():
        if source in data:
            values = data[source]
            previous = values.shift(20)
            log_change = np.log(
                values.where(values > 0) / previous.where(previous > 0)
            )
            # Choose the transform from information in this row only.  A
            # future zero/negative print must not retroactively change earlier
            # log-momentum observations into arithmetic differences.
            change = log_change.where(
                (values > 0) & (previous > 0), values - previous
            )
            output[target] = z(change.replace([np.inf, -np.inf], np.nan))

    grain_momentum = [c for c in ("wheat_momentum", "corn_momentum", "soybean_momentum") if c in output]
    if grain_momentum:
        output["grain_momentum_composite"] = output[grain_momentum].mean(axis=1)
    if {"wheat_momentum", "dxy_momentum"}.issubset(output):
        output["physical_scarcity"] = output["wheat_momentum"] + output["dxy_momentum"]
    if "breakeven_5y" in data:
        output["breakeven"] = z(data["breakeven_5y"])
    if "us_2y" in data:
        output["us2y"] = z(data["us_2y"])
    if {"grain_momentum_composite", "breakeven"}.issubset(output):
        output["food_inflation"] = output["grain_momentum_composite"] + output["breakeven"]
    return output


__all__ = [
    "build_economic_features",
    "calculate_exportable_surplus",
    "policy_stress_series",
    "rolling_input_beta",
]
