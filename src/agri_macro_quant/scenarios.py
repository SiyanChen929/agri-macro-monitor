"""Forward scenario P&L and historical portfolio stress testing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import DEFAULT_ASSET_ALIASES
from .risk import drawdown_series


DEFAULT_SCENARIOS: dict[str, dict[str, Any]] = {
    "base_case": {
        "description": "Current consensus trajectory with no discrete shock.",
        "asset_shocks": {},
        "factor_shocks": {},
    },
    "bullish_grain_shock": {
        "description": "Weather deteriorates, exporter restrictions rise, and inventories tighten.",
        "asset_shocks": {"ZW": 0.12, "ZC": 0.09, "ZS": 0.07, "CF": 0.04, "NTR": 0.03, "MOS": 0.03, "DXY": -0.01},
        "factor_shocks": {"grain": 0.10, "ag_inflation": 0.05, "fertilizer": 0.03},
    },
    "bearish_grain": {
        "description": "Rainfall and yields recover while reserves and exports normalize.",
        "asset_shocks": {"ZW": -0.10, "ZC": -0.08, "ZS": -0.06, "CF": -0.04, "NTR": -0.03, "MOS": -0.03, "DXY": 0.01},
        "factor_shocks": {"grain": -0.08, "ag_inflation": -0.04, "fertilizer": -0.03},
    },
    "stagflation": {
        "description": "Grain and oil remain elevated, breakevens rise, and policy turns hawkish.",
        "asset_shocks": {"ZW": 0.08, "ZC": 0.06, "ZS": 0.04, "NG": 0.12, "CL": 0.10, "DXY": 0.04, "US2Y": 0.03, "CF": -0.02, "NTR": -0.02, "MOS": -0.02},
        "factor_shocks": {"grain": 0.06, "energy": 0.10, "rates": 0.03, "usd": 0.04, "equity_beta": -0.05},
    },
}


DEFAULT_STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "2007-08_global_food_crisis": ("2007-01-01", "2008-06-30"),
    "2010-11_commodity_shock": ("2010-06-01", "2011-04-30"),
    "2012_us_drought": ("2012-05-01", "2012-09-30"),
    "2020_covid_supply_disruption": ("2020-02-20", "2020-05-31"),
    "2021_fertilizer_shock": ("2021-08-01", "2021-12-31"),
    "2022_russia_ukraine_invasion": ("2022-02-18", "2022-04-30"),
    "2023_black_sea_disruption": ("2023-07-10", "2023-08-31"),
}


def _weight_series(weights: pd.Series | pd.DataFrame | Mapping[str, float]) -> pd.Series:
    if isinstance(weights, pd.DataFrame):
        if {"asset", "weight"}.issubset(weights.columns):
            result = pd.to_numeric(weights["weight"], errors="coerce").groupby(weights["asset"].astype(str)).sum()
        elif weights.empty:
            result = pd.Series(dtype=float)
        else:
            result = weights.apply(pd.to_numeric, errors="coerce").iloc[-1]
    else:
        result = pd.Series(weights, dtype=float)
    result.index = result.index.astype(str)
    return pd.to_numeric(result, errors="coerce").fillna(0.0)


def _exposure_frame(
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None,
    assets: pd.Index,
) -> pd.DataFrame:
    if factor_exposures is None:
        return pd.DataFrame(index=assets)
    if isinstance(factor_exposures, Mapping):
        frame = pd.DataFrame.from_dict(factor_exposures, orient="index")
    else:
        frame = factor_exposures.copy()
    frame.index = frame.index.astype(str)
    return frame.apply(pd.to_numeric, errors="coerce").reindex(assets).fillna(0.0)


def _shock_series(shocks: Mapping[str, float], assets: pd.Index) -> pd.Series:
    raw = pd.Series(shocks, dtype=float)
    values: dict[str, float] = {}
    for asset in assets.astype(str):
        if asset in raw.index:
            values[asset] = float(raw.loc[asset])
            continue
        match = None
        for code, aliases in DEFAULT_ASSET_ALIASES.items():
            if asset == code or asset in aliases:
                match = next((candidate for candidate in (code, *aliases) if candidate in raw.index), None)
                if match is not None:
                    break
        values[asset] = float(raw.loc[match]) if match is not None else np.nan
    return pd.Series(values, dtype=float)


def run_scenario_analysis(
    weights: pd.Series | pd.DataFrame | Mapping[str, float],
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    asset_shocks: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate first-order current-portfolio P&L under forward scenarios.

    Scenario shocks are decimal returns (``0.10`` means +10%).  Asset-specific
    shocks take precedence over factor-implied shocks for the same instrument
    by default, avoiding double counting.  Set ``combine_shocks=True`` to add
    them instead.
    """

    cfg = dict(config or {})
    weight = _weight_series(weights)
    definitions: dict[str, dict[str, Any]] = {
        str(name): dict(spec) for name, spec in (scenarios or DEFAULT_SCENARIOS).items()
    }
    if asset_shocks is not None:
        if isinstance(asset_shocks, pd.DataFrame):
            for name, row in asset_shocks.iterrows():
                definitions.setdefault(str(name), {})["asset_shocks"] = row.dropna().to_dict()
        else:
            for name, shocks in asset_shocks.items():
                definitions.setdefault(str(name), {})["asset_shocks"] = dict(shocks)

    exposures = _exposure_frame(factor_exposures, weight.index)
    contributions: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    combine = bool(cfg.get("combine_shocks", False))
    for scenario, spec in definitions.items():
        explicit = _shock_series(spec.get("asset_shocks", {}), weight.index)
        factor_shock = pd.Series(spec.get("factor_shocks", {}), dtype=float)
        if exposures.empty or factor_shock.empty:
            implied = pd.Series(0.0, index=weight.index)
        else:
            implied = exposures.reindex(columns=factor_shock.index, fill_value=0.0).dot(factor_shock)
        if combine:
            total_shock = explicit.fillna(0.0) + implied
        else:
            total_shock = explicit.where(explicit.notna(), implied).fillna(0.0)
        pnl = weight * total_shock
        contribution = pd.DataFrame(
            {
                "scenario": str(scenario),
                "asset": weight.index,
                "weight": weight.to_numpy(),
                "shock": total_shock.to_numpy(),
                "pnl_contribution": pnl.to_numpy(),
            }
        )
        contributions.append(contribution)
        worst = str(pnl.idxmin()) if not pnl.empty else None
        best = str(pnl.idxmax()) if not pnl.empty else None
        rows.append(
            {
                "scenario": str(scenario),
                "portfolio_pnl": float(pnl.sum()),
                "gross_loss": float(-pnl.clip(upper=0.0).sum()),
                "gross_gain": float(pnl.clip(lower=0.0).sum()),
                "worst_contributor": worst,
                "best_contributor": best,
                "description": str(spec.get("description", "")),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("scenario")
    result.attrs["contributions"] = pd.concat(contributions, ignore_index=True) if contributions else pd.DataFrame()
    result.attrs["shock_units"] = "decimal return"
    return result


def _parse_periods(periods: Mapping[str, Any] | pd.DataFrame | None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    raw: Any = DEFAULT_STRESS_PERIODS if periods is None else periods
    result: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if isinstance(raw, pd.DataFrame):
        required = {"start", "end"}
        if not required.issubset(raw.columns):
            raise ValueError("stress-period DataFrame requires start and end columns")
        name_series = raw["name"] if "name" in raw.columns else pd.Series(raw.index, index=raw.index)
        for idx, row in raw.iterrows():
            result[str(name_series.loc[idx])] = (pd.Timestamp(row["start"]), pd.Timestamp(row["end"]))
    else:
        for name, value in raw.items():
            if isinstance(value, Mapping):
                start, end = value["start"], value["end"]
            else:
                start, end = value
            result[str(name)] = (pd.Timestamp(start), pd.Timestamp(end))
    return result


def historical_stress_test(
    weights: pd.Series | pd.DataFrame | Mapping[str, float],
    returns: pd.DataFrame,
    periods: Mapping[str, Any] | pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Replay today's static portfolio through named historical stress windows."""

    cfg = dict(config or {})
    weight = _weight_series(weights)
    frame = returns.copy()
    if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    frame.columns = frame.columns.astype(str)
    rename: dict[str, str] = {}
    for asset in weight.index:
        if asset in frame.columns:
            continue
        aliases = DEFAULT_ASSET_ALIASES.get(str(asset), (str(asset), str(asset).lower()))
        match = next((candidate for candidate in aliases if candidate in frame.columns), None)
        if match is not None:
            rename[match] = str(asset)
    frame = frame.rename(columns=rename)
    common = weight.index.intersection(frame.columns)
    weight = weight.reindex(common)
    frame = frame.reindex(columns=common)
    if common.empty:
        raise ValueError("weights and historical returns have no common assets")

    parsed = _parse_periods(periods)
    include_unavailable = bool(cfg.get("include_unavailable", False))
    backtest_cfg = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), Mapping) else {}
    annualization = float(
        cfg.get(
            "annualization",
            cfg.get("annualization_days", backtest_cfg.get("annualization", backtest_cfg.get("annualization_days", 252))),
        )
    )
    minimum_priced_coverage = float(
        np.clip(cfg.get("min_priced_weight_coverage", 1.0), 0.0, 1.0)
    )
    rows: list[dict[str, Any]] = []
    shocks: list[pd.DataFrame] = []
    for name, (start, end) in parsed.items():
        sample = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if sample.empty and not include_unavailable:
            continue
        if sample.empty:
            rows.append(
                {
                    "stress_period": name,
                    "start": start,
                    "end": end,
                    "n_observations": 0,
                    "portfolio_pnl": np.nan,
                    "worst_daily_pnl": np.nan,
                    "maximum_drawdown": np.nan,
                    "annualized_volatility": np.nan,
                    "mean_priced_weight_coverage": np.nan,
                    "minimum_priced_weight_coverage": np.nan,
                    "excluded_unpriced_days": 0,
                }
            )
            continue
        active_assets = weight.index[weight.abs().gt(1e-12)]
        active_sample = sample.reindex(columns=active_assets)
        active_weight = weight.reindex(active_assets)
        gross_weight = float(active_weight.abs().sum())
        if gross_weight > 0.0:
            priced_coverage = active_sample.notna().mul(
                active_weight.abs(), axis=1
            ).sum(axis=1) / gross_weight
        else:
            priced_coverage = pd.Series(1.0, index=sample.index)
        eligible = priced_coverage.ge(minimum_priced_coverage - 1e-12)
        clean = active_sample.loc[eligible].dropna(how="any")
        excluded_days = int(len(sample) - len(clean))
        if clean.empty:
            rows.append(
                {
                    "stress_period": name,
                    "start": sample.index.min(),
                    "end": sample.index.max(),
                    "n_observations": 0,
                    "portfolio_pnl": np.nan,
                    "worst_daily_pnl": np.nan,
                    "maximum_drawdown": np.nan,
                    "annualized_volatility": np.nan,
                    "mean_priced_weight_coverage": float(priced_coverage.mean()),
                    "minimum_priced_weight_coverage": float(priced_coverage.min()),
                    "excluded_unpriced_days": excluded_days,
                }
            )
            continue
        portfolio_returns = clean.mul(active_weight, axis=1).sum(axis=1)
        equity = (1.0 + portfolio_returns).cumprod()
        drawdown = drawdown_series(portfolio_returns, "returns")
        asset_shock = (1.0 + clean).prod(axis=0) - 1.0
        shock_frame = asset_shock.rename("shock").reset_index().rename(columns={"index": "asset"})
        shock_frame.attrs = {}
        shock_frame.insert(0, "stress_period", name)
        shocks.append(shock_frame)
        rows.append(
            {
                "stress_period": name,
                "start": sample.index.min(),
                "end": sample.index.max(),
                "n_observations": int(len(sample)),
                "priced_observations": int(len(clean)),
                "portfolio_pnl": float(equity.iloc[-1] - 1.0),
                "worst_daily_pnl": float(portfolio_returns.min()),
                "maximum_drawdown": float(drawdown.min()),
                "annualized_volatility": float(portfolio_returns.std(ddof=1) * np.sqrt(annualization)) if len(clean) > 1 else 0.0,
                "mean_priced_weight_coverage": float(priced_coverage.mean()),
                "minimum_priced_weight_coverage": float(priced_coverage.min()),
                "excluded_unpriced_days": excluded_days,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("stress_period")
    result.attrs["asset_shocks"] = pd.concat(shocks, ignore_index=True) if shocks else pd.DataFrame()
    return result


# Familiar aliases.
scenario_analysis = run_scenario_analysis
stress_test = historical_stress_test


__all__ = [
    "DEFAULT_SCENARIOS",
    "DEFAULT_STRESS_PERIODS",
    "historical_stress_test",
    "run_scenario_analysis",
    "scenario_analysis",
    "stress_test",
]
