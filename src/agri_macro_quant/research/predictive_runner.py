"""Integrated, artifact-producing runner for the frozen grain experiment.

The statistical machinery lives in :mod:`predictive_experiment`; this module
owns configuration loading, point-in-time feature alignment, the two
predeclared continuous-proxy treatments, a close-to-close implementation
backtest, and an intentionally conservative report.  It never changes a model
after looking at holdout results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .predictive_data import PredictiveDataBundle, acquire_predictive_data
from .predictive_experiment import (
    CORE_FEATURE_COLUMNS,
    PROXY_CONTRACT_MONTHS,
    RIDGE_CORE,
    RIDGE_CORE_BOTH,
    RIDGE_CORE_USDA,
    RIDGE_CORE_WEATHER,
    PredictiveExperimentConfig,
    PredictiveExperimentResult,
    build_daily_prediction_panel,
    development_selection_plan,
    make_crop_year_folds,
    run_predictive_experiment,
    sample_weekly_panel,
)


USDA_REVISION_COLUMNS: tuple[str, ...] = (
    "us_production_revision",
    "us_yield_revision",
    "us_ending_stocks_revision",
    "world_production_revision",
    "world_ending_stocks_revision",
    "us_stocks_to_use_revision",
)
USDA_LEVEL_COLUMNS: tuple[str, ...] = ("us_stocks_to_use",)
WEATHER_FEATURE_COLUMNS: tuple[str, ...] = (
    "temperature_anomaly_14d",
    "precipitation_anomaly_30d",
    "heat_stress_days_14d",
    "growing_season_temperature_anomaly",
    "growing_season_precipitation_anomaly",
)

EXPECTED_HOLDOUT_MODELS: tuple[str, ...] = (
    "historical_mean",
    "seasonality",
    "momentum_60",
    "ridge_core",
    "development_selected_ridge",
)
EXPECTED_RIDGE_CANDIDATES: tuple[str, ...] = (
    RIDGE_CORE_USDA,
    RIDGE_CORE_WEATHER,
    RIDGE_CORE_BOTH,
)


@dataclass(frozen=True)
class PortfolioCostConfig:
    """Frozen implementation assumptions from the experiment YAML."""

    maximum_asset_weight: float
    maximum_gross_exposure: float
    transaction_cost_bps_one_way: float
    slippage_bps_one_way: float
    commission_bps_one_way: float
    futures_roll_cost_bps: float
    cost_sensitivity_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class LoadedPredictiveConfig:
    """Resolved statistical and implementation specification."""

    experiment_id: str
    status: str
    start_date: pd.Timestamp
    as_of_date: pd.Timestamp
    statistical: PredictiveExperimentConfig
    portfolio: PortfolioCostConfig
    raw: Mapping[str, Any]
    path: Path
    sha256: str


@dataclass(frozen=True)
class PredictiveRunArtifacts:
    """Materialized output from one indivisible experiment execution."""

    output_dir: Path
    primary: PredictiveExperimentResult
    raw_sensitivity: PredictiveExperimentResult
    portfolio_daily: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    conclusion: str
    report_markdown: Path
    report_html: Path
    manifest: Path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash values, index, columns, and dtypes for a resolved input frame."""

    digest = sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "index_names": [str(name) for name in frame.index.names],
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_predictive_config(path: str | Path) -> LoadedPredictiveConfig:
    """Load and validate the committed preregistration without mutating it."""

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("predictive experiment config must be a YAML mapping")
    experiment = payload.get("experiment", {})
    target = payload.get("target", {})
    validation = payload.get("validation", {})
    data_contract = payload.get("data_contract", {})
    portfolio = payload.get("portfolio", {})
    claim_policy = payload.get("claim_policy", {})
    assets = tuple(str(value).upper() for value in experiment.get("assets", {}))
    if assets != ("ZW", "ZC", "ZS"):
        raise ValueError("frozen v1 universe must be ordered as ZW, ZC, ZS")
    if int(target.get("execution_lag_sessions", -1)) != 1:
        raise ValueError("frozen experiment requires a one-session execution lag")
    if not bool(claim_policy.get("do_not_tune_after_holdout_open", False)):
        raise ValueError("claim policy must prohibit tuning after the holdout opens")
    expected_support_gate = {
        "primary_holdout_oos_r2_strictly_above": 0.0,
        "raw_sensitivity_holdout_oos_r2_strictly_above": 0.0,
        "primary_holdout_1x_annualized_net_return_strictly_above": 0.0,
    }
    if claim_policy.get("exploratory_support_gate") != expected_support_gate:
        raise ValueError("claim_policy exploratory support gate does not match execution")
    weather_contract = data_contract.get("weather", {})
    if int(weather_contract.get("assumed_publication_lag_calendar_days", -1)) != 3:
        raise ValueError("frozen experiment requires the conservative three-day weather lag")

    feature_families = payload.get("feature_families", {})
    expected_families = {
        "core": [column.removeprefix("core__") for column in CORE_FEATURE_COLUMNS],
        "usda": [*USDA_REVISION_COLUMNS, *USDA_LEVEL_COLUMNS, "days_since_usda_release"],
        "weather": list(WEATHER_FEATURE_COLUMNS),
    }
    if feature_families != expected_families:
        raise ValueError(
            "feature_families must exactly match the executable v1 feature registry"
        )
    if tuple(validation.get("holdout_models", ())) != EXPECTED_HOLDOUT_MODELS:
        raise ValueError("validation.holdout_models does not match the executable holdout roster")
    selection = payload.get("selection_rule", {})
    expected_selection = {
        "baseline_ridge_family": RIDGE_CORE,
        "proxy_variant_selection_scope": (
            "primary_development_locked_for_all_variants"
        ),
        "candidate_order": list(EXPECTED_RIDGE_CANDIDATES),
        "require_positive_pooled_oos_r2_improvement": True,
        "require_asset_improvement_count": 2,
        "require_majority_crop_year_fold_improvement": True,
        "otherwise_select": RIDGE_CORE,
    }
    if selection != expected_selection:
        raise ValueError("selection_rule does not match the executable development gate")
    if portfolio.get("holdout_calendar") != "common_intersection_all_assets":
        raise ValueError("portfolio holdout calendar must be the all-asset intersection")
    if portfolio.get("roll_cost_timing") != "exact_preregistered_anchor_sessions":
        raise ValueError("portfolio roll costs must use exact anchor sessions")
    roll_rule = data_contract.get("market", {}).get("roll_gap_rule", {})
    if roll_rule.get("session_rule") != "first_valid_session_on_or_after":
        raise ValueError("frozen proxy repair requires the first-valid-session rule")
    if float(roll_rule.get("replacement_return", np.nan)) != 0.0:
        raise ValueError("frozen proxy repair requires a zero anchor-session return")
    configured_contract_months = {
        asset: tuple(roll_rule.get(f"{asset}_contract_months", ()))
        for asset in assets
    }
    expected_contract_months = {
        asset: tuple(PROXY_CONTRACT_MONTHS[asset]) for asset in assets
    }
    if configured_contract_months != expected_contract_months:
        raise ValueError("roll_gap_rule contract months do not match the executable registry")

    statistical = PredictiveExperimentConfig(
        assets=assets,
        horizon_sessions=int(target["horizon_sessions"]),
        weekly_rule=str(target["decision_frequency"]),
        minimum_train_crop_years=int(validation["minimum_training_crop_years"]),
        minimum_weekly_observations_per_crop_year=int(
            validation["minimum_weekly_observations_per_crop_year"]
        ),
        embargo_business_days=int(validation["embargo_business_days"]),
        holdout_crop_years=int(validation["locked_holdout_crop_years"]),
        winsor_lower=_finite_float(
            validation["winsor_lower_quantile"], name="winsor_lower_quantile"
        ),
        winsor_upper=_finite_float(
            validation["winsor_upper_quantile"], name="winsor_upper_quantile"
        ),
        minimum_feature_coverage=_finite_float(
            validation["minimum_feature_coverage"], name="minimum_feature_coverage"
        ),
        neutralize_proxy_roll_gaps=True,
        proxy_roll_anchor_day=int(roll_rule["anchor_day"]),
        proxy_roll_maximum_day=int(roll_rule["maximum_calendar_day"]),
        alphas=tuple(float(value) for value in validation["ridge_alphas"]),
    )
    cost_multipliers = tuple(
        _finite_float(value, name="cost_sensitivity_multiplier")
        for value in portfolio["cost_sensitivity_multipliers"]
    )
    if cost_multipliers != (1.0, 1.5, 2.0):
        raise ValueError("frozen cost sensitivities must be exactly 1.0x, 1.5x, 2.0x")
    costs = PortfolioCostConfig(
        maximum_asset_weight=_finite_float(
            portfolio["maximum_asset_weight"], name="maximum_asset_weight"
        ),
        maximum_gross_exposure=_finite_float(
            portfolio["maximum_gross_exposure"], name="maximum_gross_exposure"
        ),
        transaction_cost_bps_one_way=_finite_float(
            portfolio["transaction_cost_bps_one_way"],
            name="transaction_cost_bps_one_way",
        ),
        slippage_bps_one_way=_finite_float(
            portfolio["slippage_bps_one_way"], name="slippage_bps_one_way"
        ),
        commission_bps_one_way=_finite_float(
            portfolio["commission_bps_one_way"], name="commission_bps_one_way"
        ),
        futures_roll_cost_bps=_finite_float(
            portfolio["futures_roll_cost_bps"], name="futures_roll_cost_bps"
        ),
        cost_sensitivity_multipliers=cost_multipliers,
    )
    if costs.maximum_asset_weight <= 0.0 or costs.maximum_gross_exposure <= 0.0:
        raise ValueError("portfolio exposure limits must be positive")
    return LoadedPredictiveConfig(
        experiment_id=str(experiment["id"]),
        status=str(experiment["status"]),
        start_date=pd.Timestamp(experiment["start_date"]).normalize(),
        as_of_date=pd.Timestamp(experiment["as_of_date"]).normalize(),
        statistical=statistical,
        portfolio=costs,
        raw=payload,
        path=source,
        sha256=_file_sha256(source),
    )


def _valid_asset_sessions(market_prices: pd.DataFrame, asset: str) -> pd.DatetimeIndex:
    if asset not in market_prices:
        raise KeyError(f"market price panel has no {asset} column")
    dates = pd.DatetimeIndex(
        pd.to_datetime(market_prices.index[market_prices[asset].notna()], errors="raise")
    )
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    return dates.sort_values().unique()


def _first_session_strictly_after(
    available_date: pd.Timestamp, sessions: pd.DatetimeIndex
) -> pd.Timestamp | pd.NaT:
    position = int(sessions.searchsorted(pd.Timestamp(available_date).normalize(), side="right"))
    if position >= len(sessions):
        return pd.NaT
    return pd.Timestamp(sessions[position])


def align_usda_to_market(
    usda_features: pd.DataFrame,
    market_prices: pd.DataFrame,
    *,
    assets: Sequence[str] = ("ZW", "ZC", "ZS"),
    weekly_rule: str = "W-FRI",
) -> pd.DataFrame:
    """Create causal daily USDA inputs for the core's exact-date join.

    Revision surprises are zero outside their activation week and persist from
    the first tradable session through that week's final market decision.  The
    stocks-to-use level is carried from the most recently tradable release and
    accompanied by its age.  Availability is strict: a CSV dated ``t`` first
    enters on the first valid asset session *after* ``t``.
    """

    required = {"available_date", "release_date", "asset"}
    missing = required.difference(usda_features.columns)
    if missing:
        raise ValueError(f"USDA feature frame is missing {sorted(missing)}")
    raw = usda_features.copy()
    raw["available_date"] = pd.to_datetime(raw["available_date"], errors="raise").dt.normalize()
    raw["release_date"] = pd.to_datetime(raw["release_date"], errors="raise").dt.normalize()
    raw["asset"] = raw["asset"].astype(str).str.upper()
    panels: list[pd.DataFrame] = []
    for asset in assets:
        sessions = _valid_asset_sessions(market_prices, str(asset))
        if sessions.empty:
            continue
        releases = raw.loc[raw["asset"].eq(str(asset))].copy()
        releases["activation_date"] = releases["available_date"].map(
            lambda value: _first_session_strictly_after(pd.Timestamp(value), sessions)
        )
        releases = releases.dropna(subset=["activation_date"])
        releases = releases.sort_values(
            ["activation_date", "release_date"], kind="stable"
        ).drop_duplicates("activation_date", keep="last")

        aligned = pd.DataFrame({"date": sessions, "asset": str(asset)})
        revision_columns = [column for column in USDA_REVISION_COLUMNS if column in releases]
        level_columns = [column for column in USDA_LEVEL_COLUMNS if column in releases]
        for column in revision_columns:
            aligned[column] = 0.0
        if not releases.empty:
            session_period = pd.Series(sessions.to_period(weekly_rule), index=aligned.index)
            for release in releases.itertuples(index=False):
                activation = pd.Timestamp(release.activation_date)
                active = (sessions >= activation) & session_period.eq(
                    activation.to_period(weekly_rule)
                ).to_numpy(dtype=bool)
                for column in revision_columns:
                    value = pd.to_numeric(
                        pd.Series([getattr(release, column)]), errors="coerce"
                    ).iloc[0]
                    aligned.loc[active, column] = float(value) if pd.notna(value) else np.nan

            right_columns = [
                "activation_date",
                "release_date",
                "available_date",
                *level_columns,
            ]
            release_levels = releases[right_columns].copy().sort_values("activation_date")
            aligned = pd.merge_asof(
                aligned.sort_values("date"),
                release_levels,
                left_on="date",
                right_on="activation_date",
                direction="backward",
                allow_exact_matches=True,
            )
            aligned["days_since_usda_release"] = (
                aligned["date"] - pd.to_datetime(aligned["activation_date"])
            ).dt.days.astype(float)
        else:
            for column in level_columns:
                aligned[column] = np.nan
            aligned["activation_date"] = pd.NaT
            aligned["release_date"] = pd.NaT
            aligned["available_date"] = pd.NaT
            aligned["days_since_usda_release"] = np.nan
        panels.append(aligned)
    if not panels:
        raise ValueError("USDA alignment produced no asset panels")
    result = pd.concat(panels, ignore_index=True).sort_values(["date", "asset"])
    keep = [
        "date",
        "asset",
        *[column for column in USDA_REVISION_COLUMNS if column in result],
        *[column for column in USDA_LEVEL_COLUMNS if column in result],
        "days_since_usda_release",
    ]
    return result.reindex(columns=keep).reset_index(drop=True)


def align_weather_to_market(
    weather_features: pd.DataFrame,
    market_prices: pd.DataFrame,
    *,
    assets: Sequence[str] = ("ZW", "ZC", "ZS"),
    publication_lag_days: int = 3,
) -> pd.DataFrame:
    """Merge the latest lagged NOAA observation into each market session."""

    required = {"observation_date", "asset"}
    missing = required.difference(weather_features.columns)
    if missing:
        raise ValueError(f"weather feature frame is missing {sorted(missing)}")
    raw = weather_features.copy()
    raw["observation_date"] = pd.to_datetime(
        raw["observation_date"], errors="raise"
    ).dt.normalize()
    # Reconstruct availability from observation date instead of trusting a
    # caller-provided field.  That makes the frozen lag auditable and prevents
    # an accidentally contemporaneous weather join.
    raw["available_date"] = raw["observation_date"] + pd.Timedelta(
        days=int(publication_lag_days)
    )
    raw["asset"] = raw["asset"].astype(str).str.upper()
    feature_columns = [column for column in WEATHER_FEATURE_COLUMNS if column in raw]
    if not feature_columns:
        raise ValueError("weather frame contains none of the preregistered features")
    panels: list[pd.DataFrame] = []
    for asset in assets:
        sessions = _valid_asset_sessions(market_prices, str(asset))
        if sessions.empty:
            continue
        observations = (
            raw.loc[raw["asset"].eq(str(asset)), [
                "available_date",
                "observation_date",
                *feature_columns,
            ]]
            .sort_values(["available_date", "observation_date"], kind="stable")
            .drop_duplicates("available_date", keep="last")
        )
        left = pd.DataFrame({"date": sessions, "asset": str(asset)})
        aligned = pd.merge_asof(
            left.sort_values("date"),
            observations,
            left_on="date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        panels.append(aligned)
    if not panels:
        raise ValueError("weather alignment produced no asset panels")
    result = pd.concat(panels, ignore_index=True).sort_values(["date", "asset"])
    keep = ["date", "asset", *feature_columns]
    return result.reindex(columns=keep).reset_index(drop=True)


def _fold_training_target_volatility(
    result: PredictiveExperimentResult,
) -> Mapping[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for fold in result.folds.itertuples(index=False):
        asset = str(fold.asset)
        frame = result.weekly_panel.xs(asset, level="asset").sort_index()
        years = tuple(int(value) for value in str(fold.train_crop_years).split(",") if value)
        target_end = pd.to_datetime(frame["target_end_date"], errors="coerce")
        eligible = frame.loc[
            frame["crop_year"].astype(int).isin(years)
            & frame["target_return_21d"].notna()
            & target_end.lt(pd.Timestamp(fold.purge_cutoff))
        ]
        volatility = float(
            pd.to_numeric(eligible["target_return_21d"], errors="coerce").std(ddof=0)
        )
        output[(asset, int(fold.test_crop_year))] = (
            volatility if np.isfinite(volatility) and volatility > 1.0e-12 else np.nan
        )
    return output


def _selected_signals(
    result: PredictiveExperimentResult,
    *,
    split: str = "holdout",
) -> pd.DataFrame:
    selected = result.predictions.loc[
        result.predictions["model"].eq(result.selected_model)
        & result.predictions["split"].eq(str(split))
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no {split} predictions exist for {result.selected_model}"
        )
    volatility = _fold_training_target_volatility(result)
    selected["training_target_volatility"] = [
        volatility.get((str(asset), int(year)), np.nan)
        for asset, year in zip(selected["asset"], selected["crop_year"], strict=True)
    ]
    scale = selected["prediction"] / selected["training_target_volatility"]
    selected["unconstrained_signal"] = np.tanh(scale)
    return selected.sort_values(["date", "asset"]).reset_index(drop=True)


def build_costed_portfolio(
    result: PredictiveExperimentResult,
    costs: PortfolioCostConfig,
    *,
    variant: str,
    split: str = "holdout",
    target_policy: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate OOS forecasts into a lagged daily path and cost stresses.

    Only close prices are available.  A Friday decision changes the target at
    the following asset session's close; P&L therefore begins on the subsequent
    close-to-close interval.  This is conservative and avoids crediting the
    unobserved next-session open-to-close return.
    """

    signals = _selected_signals(result, split=split)
    first_by_asset = signals.groupby("asset")["date"].min()
    last_by_asset = signals.groupby("asset")["date"].max()
    missing_assets = set(result.config.assets).difference(first_by_asset.index)
    if missing_assets:
        raise ValueError(f"portfolio split lacks forecasts for {sorted(missing_assets)}")
    common_first_decision = pd.Timestamp(first_by_asset.max())
    common_last_decision = pd.Timestamp(last_by_asset.min())
    if common_last_decision < common_first_decision:
        raise ValueError("asset-specific forecast windows have no common intersection")
    signals = signals.loc[
        signals["date"].between(common_first_decision, common_last_decision)
    ].copy()
    signals["target_weight"] = signals["unconstrained_signal"].clip(
        -costs.maximum_asset_weight, costs.maximum_asset_weight
    )
    if target_policy == "cost_aware_band":
        from .cost_aware_execution import apply_cost_aware_targets

        signals = apply_cost_aware_targets(
            signals,
            one_way_cost_bps=(costs.transaction_cost_bps_one_way
                              + costs.slippage_bps_one_way
                              + costs.commission_bps_one_way),
            maximum_asset_weight=costs.maximum_asset_weight,
            target_sessions=result.config.horizon_sessions,
            decision_sessions=5,
        )
    elif target_policy != "baseline":
        raise ValueError(f"unknown target policy: {target_policy}")
    daily_columns = ["date", "asset", "price"]
    if "proxy_roll_gap_flag" in result.daily_panel.columns:
        daily_columns.append("proxy_roll_gap_flag")
    daily = result.daily_panel.reset_index()[daily_columns].copy()
    if "proxy_roll_gap_flag" not in daily:
        daily["proxy_roll_gap_flag"] = False
    daily["proxy_roll_gap_flag"] = (
        daily["proxy_roll_gap_flag"].fillna(False).astype(bool)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["asset", "date"])
    assembled: list[pd.DataFrame] = []
    for asset in result.config.assets:
        prices = daily.loc[
            daily["asset"].eq(asset) & daily["price"].notna()
        ].sort_values("date")
        asset_signals = signals.loc[signals["asset"].eq(asset)].copy()
        if prices.empty or asset_signals.empty:
            continue
        first_decision = pd.Timestamp(asset_signals["date"].min())
        last_decision = pd.Timestamp(asset_signals["date"].max())
        # The final evaluated decision is closed after one normal rebalance
        # interval instead of being carried through unevaluated future data.
        nominal_path_end = last_decision + pd.Timedelta(days=7)
        future_sessions = pd.DatetimeIndex(prices["date"])
        close_position = int(future_sessions.searchsorted(nominal_path_end, side="right"))
        path_end = (
            pd.Timestamp(future_sessions[close_position])
            if close_position < len(future_sessions)
            else pd.Timestamp(future_sessions[-1])
        )
        prices = prices.loc[
            prices["date"].ge(first_decision) & prices["date"].le(path_end)
        ].copy()
        merged = pd.merge_asof(
            prices,
            asset_signals[["date", "target_weight"]].rename(
                columns={"date": "decision_date"}
            ),
            left_on="date",
            right_on="decision_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged["target_weight"] = merged["target_weight"].fillna(0.0)
        if path_end > nominal_path_end:
            merged.loc[merged["date"].eq(path_end), "target_weight"] = 0.0
        # At t the target is executed at the close; the old position earns the
        # return into t and the new position begins earning at t+1.
        merged["held_weight"] = merged["target_weight"].shift(1).fillna(0.0)
        merged["asset_return"] = pd.to_numeric(
            merged["price"], errors="coerce"
        ).pct_change(fill_method=None)
        merged["weight_turnover"] = merged["target_weight"].diff().abs()
        merged.loc[merged.index[0], "weight_turnover"] = abs(
            float(merged.loc[merged.index[0], "target_weight"])
        )
        assembled.append(
            merged[[
                "date",
                "asset",
                "price",
                "asset_return",
                "decision_date",
                "target_weight",
                "held_weight",
                "weight_turnover",
                "proxy_roll_gap_flag",
            ]]
        )
    if not assembled:
        raise ValueError("selected forecasts produced no tradable daily path")
    long = pd.concat(assembled, ignore_index=True).sort_values(["date", "asset"])
    weight = long.pivot(index="date", columns="asset", values="held_weight").fillna(0.0)
    target_weight = long.pivot(
        index="date", columns="asset", values="target_weight"
    ).fillna(0.0)
    returns = long.pivot(index="date", columns="asset", values="asset_return")
    roll_flags = long.pivot(
        index="date", columns="asset", values="proxy_roll_gap_flag"
    ).fillna(False).astype(bool)
    common_dates = weight.index.union(returns.index).sort_values()
    weight = weight.reindex(common_dates).fillna(0.0)
    target_weight = target_weight.reindex(common_dates).fillna(0.0)
    returns = returns.reindex(common_dates)
    roll_flags = roll_flags.reindex(common_dates).fillna(False).astype(bool)
    gross = weight.abs().sum(axis=1)
    over_limit = gross.gt(costs.maximum_gross_exposure)
    if over_limit.any():
        scaler = costs.maximum_gross_exposure / gross.loc[over_limit]
        weight.loc[over_limit] = weight.loc[over_limit].mul(scaler, axis=0)
        target_weight.loc[over_limit] = target_weight.loc[over_limit].mul(scaler, axis=0)
    gross_exposure = weight.abs().sum(axis=1)
    turnover = target_weight.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = target_weight.iloc[0].abs().sum()
    gross_return = (weight * returns.fillna(0.0)).sum(axis=1)
    base_one_way_bps = (
        costs.transaction_cost_bps_one_way
        + costs.slippage_bps_one_way
        + costs.commission_bps_one_way
    )
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for multiplier in costs.cost_sensitivity_multipliers:
        transaction_cost = turnover * (base_one_way_bps / 10_000.0) * multiplier
        roll_cost = (
            weight.abs().where(roll_flags, 0.0).sum(axis=1)
            * (costs.futures_roll_cost_bps / 10_000.0)
            * multiplier
        )
        net_return = gross_return - transaction_cost - roll_cost
        gross_nav = (1.0 + gross_return).cumprod()
        net_nav = (1.0 + net_return).cumprod()
        net_drawdown = net_nav / net_nav.cummax().clip(lower=1.0) - 1.0
        frame = pd.DataFrame(
            {
                "date": common_dates,
                "variant": variant,
                "evaluation_split": str(split),
                "common_first_decision": common_first_decision,
                "common_last_decision": common_last_decision,
                "selected_model": result.selected_model,
                "cost_multiplier": float(multiplier),
                "gross_return": gross_return.to_numpy(dtype=float),
                "turnover": turnover.to_numpy(dtype=float),
                "gross_exposure": gross_exposure.to_numpy(dtype=float),
                "transaction_cost": transaction_cost.to_numpy(dtype=float),
                "roll_cost": roll_cost.to_numpy(dtype=float),
                "net_return": net_return.to_numpy(dtype=float),
                "gross_nav": gross_nav.to_numpy(dtype=float),
                "net_nav": net_nav.to_numpy(dtype=float),
                "net_drawdown": net_drawdown.to_numpy(dtype=float),
            }
        )
        rows.append(frame)
        observations = int(net_return.notna().sum())
        annual_return = (
            float(net_nav.iloc[-1] ** (252.0 / observations) - 1.0)
            if observations > 0 and float(net_nav.iloc[-1]) > 0.0
            else np.nan
        )
        annual_volatility = float(net_return.std(ddof=0) * np.sqrt(252.0))
        metric_rows.append(
            {
                "variant": variant,
                "evaluation_split": str(split),
                "common_first_decision": common_first_decision,
                "common_last_decision": common_last_decision,
                "selected_model": result.selected_model,
                "cost_multiplier": float(multiplier),
                "observations": observations,
                "annualized_net_return": annual_return,
                "annualized_arithmetic_net_return": float(net_return.mean() * 252.0),
                "annualized_net_volatility": annual_volatility,
                "net_sharpe_zero_rate": float(net_return.mean() * 252.0) / annual_volatility
                if annual_volatility > 0.0
                else np.nan,
                "geometric_return_to_volatility": annual_return / annual_volatility
                if annual_volatility > 0.0
                else np.nan,
                "maximum_drawdown": float(net_drawdown.min()),
                "average_gross_exposure": float(gross_exposure.mean()),
                "annualized_turnover": float(turnover.mean() * 252.0),
                "total_transaction_cost": float(transaction_cost.sum()),
                "total_roll_cost": float(roll_cost.sum()),
                "terminal_net_nav": float(net_nav.iloc[-1]),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metric_rows)


def _metric_value(
    table: pd.DataFrame, split: str, model: str, column: str
) -> float:
    rows = table.loc[table["split"].eq(split) & table["model"].eq(model), column]
    return float(rows.iloc[0]) if len(rows) else np.nan


def _experiment_conclusion(
    primary: PredictiveExperimentResult,
    raw_sensitivity: PredictiveExperimentResult,
    portfolio_metrics: pd.DataFrame,
) -> tuple[str, str]:
    primary_r2 = _metric_value(
        primary.metrics_overall,
        "holdout",
        primary.selected_model,
        "oos_r2_vs_historical_mean",
    )
    raw_r2 = _metric_value(
        raw_sensitivity.metrics_overall,
        "holdout",
        raw_sensitivity.selected_model,
        "oos_r2_vs_historical_mean",
    )
    cost_row = portfolio_metrics.loc[
        portfolio_metrics["variant"].eq("delivery_anchor_neutralized_primary")
        & portfolio_metrics["evaluation_split"].eq("holdout")
        & portfolio_metrics["cost_multiplier"].eq(1.0)
    ]
    net_return = (
        float(cost_row.iloc[0]["annualized_net_return"]) if len(cost_row) else np.nan
    )
    same_family = primary.selected_model == raw_sensitivity.selected_model
    if same_family and all(
        np.isfinite(value) and value > 0.0
        for value in (primary_r2, raw_r2, net_return)
    ):
        return (
            "exploratory_support_only",
            "The predeclared checks are directionally positive, but this is not a validated "
            "predictive edge: vendor continuous proxies and non-vintage weather still require "
            "an untouched contract-level confirmation.",
        )
    return (
        "predictive_edge_not_supported",
        "The experiment does not support a robust predictive edge under the predeclared "
        "holdout, raw-stitch sensitivity, and 1.0x implementation-cost checks.",
    )


def _display_metrics(result: PredictiveExperimentResult) -> pd.DataFrame:
    columns = [
        "split",
        "model",
        "observations",
        "oos_r2_vs_historical_mean",
        "rmse",
        "spearman_ic",
        "directional_accuracy",
    ]
    frame = result.metrics_overall.reindex(columns=columns).copy()
    numeric = [column for column in columns if column not in {"split", "model"}]
    frame[numeric] = frame[numeric].round(4)
    return frame


def _markdown_report(
    loaded: LoadedPredictiveConfig,
    bundle: PredictiveDataBundle,
    primary: PredictiveExperimentResult,
    raw_sensitivity: PredictiveExperimentResult,
    portfolio_metrics: pd.DataFrame,
    conclusion_code: str,
    conclusion_text: str,
) -> str:
    primary_metric = _display_metrics(primary)
    raw_metric = _display_metrics(raw_sensitivity)
    cost_metric = portfolio_metrics.copy()
    numeric = cost_metric.select_dtypes(include=[np.number]).columns
    cost_metric[numeric] = cost_metric[numeric].round(4)
    lineage = bundle.lineage.copy()
    lineage_columns = [
        column
        for column in ("dataset", "provider", "quality", "rows", "first_date", "last_date", "failures")
        if column in lineage
    ]
    return "\n".join(
        [
            f"# Predictive experiment: {loaded.experiment_id}",
            "",
            f"**Outcome: `{conclusion_code}`.** {conclusion_text}",
            "",
            "This was one execution of the frozen exploratory specification. Candidate "
            "families were compared only on expanding development folds; each predeclared "
            "proxy treatment then opened its latest three complete crop-year holdouts once. "
            "No model or alpha was changed after those results were observed.",
            "",
            "## What ran",
            "",
            f"- As-of date: {loaded.as_of_date.date().isoformat()}",
            f"- Assets: {', '.join(loaded.statistical.assets)}",
            f"- Target: next-session through {loaded.statistical.horizon_sessions}-session return",
            f"- Primary selected family: `{primary.selected_model}`",
            f"- Family locked into raw-stitch sensitivity: `{raw_sensitivity.selected_model}`",
            f"- Raw-only development preference (diagnostic, not used): `{raw_sensitivity.development_selected_model}`",
            "- Decisions: final priced session in each W-FRI period",
            "- Validation: expanding crop-year folds, label-overlap purge, five-business-day embargo",
            "- Portfolio window: common intersection of all three asset holdouts",
            "",
            "## Primary delivery-anchor-neutralized proxy metrics",
            "",
            primary_metric.to_markdown(index=False),
            "",
            "## Raw vendor-stitch sensitivity",
            "",
            raw_metric.to_markdown(index=False),
            "",
            "## Selected-forecast implementation path",
            "",
            cost_metric.to_markdown(index=False),
            "",
            "Signals use `tanh(forecast / training-target-volatility)`, a one-third asset "
            "cap, and a 1.0 gross cap. With close-only data, the next-session close is treated "
            "as execution and P&L begins on the following close-to-close interval. Transaction "
            "and slippage costs are charged on position changes. The 3 bps roll charge is "
            "applied to absolute held exposure on each preregistered delivery-month anchor "
            "session; it is still a proxy, not observed two-leg roll P&L.",
            "",
            "## Point-in-time controls",
            "",
            "- Historical USDA CSV values become eligible one calendar day after report date "
            "and activate only on the first asset market session strictly after that date.",
            "- USDA revisions are event features active only through the activation week's "
            "decision. The latest stocks-to-use level is carried causally with days-since-release.",
            "- NOAA observations are reconstructed with a three-calendar-day lag and joined "
            "backward; no market decision can use a later weather observation.",
            "- Winsorization, missing-value handling, standardization, alpha choice, and label "
            "purging use training data only.",
            "",
            "## Data lineage",
            "",
            lineage.reindex(columns=lineage_columns).to_markdown(index=False),
            "",
            "## Limits and claim boundary",
            "",
            "This is a falsification test, not a deployable trading model. Yahoo's generic "
            "series are vendor-stitched rather than individual contracts; the NOAA endpoint "
            "files are current quality-controlled histories rather than archived daily vintages; "
            "and overlapping 21-session labels reduce the effective independent sample size. "
            "A positive exploratory result still requires individual-contract settlements, a "
            "causal roll implementation, dependence-aware uncertainty, and a newly untouched "
            "holdout before any predictive or tradability claim.",
            "",
        ]
    )


def _html_report(
    loaded: LoadedPredictiveConfig,
    bundle: PredictiveDataBundle,
    primary: PredictiveExperimentResult,
    raw_sensitivity: PredictiveExperimentResult,
    portfolio_metrics: pd.DataFrame,
    conclusion_code: str,
    conclusion_text: str,
) -> str:
    primary_table = _display_metrics(primary).to_html(index=False, border=0, classes="metrics")
    raw_table = _display_metrics(raw_sensitivity).to_html(index=False, border=0, classes="metrics")
    cost_table = portfolio_metrics.round(4).to_html(index=False, border=0, classes="metrics")
    lineage_columns = [
        column
        for column in ("dataset", "provider", "quality", "rows", "first_date", "last_date", "failures")
        if column in bundle.lineage
    ]
    lineage_table = bundle.lineage.reindex(columns=lineage_columns).to_html(
        index=False, border=0, classes="metrics"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(loaded.experiment_id)} predictive experiment</title>
<style>
body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:32px;color:#172033;background:#f7f9fc;line-height:1.55}}
h1,h2{{color:#10264d}} .outcome{{padding:18px;border-left:5px solid #b45b26;background:#fff3e8}}
.card{{background:white;padding:20px;margin:18px 0;border-radius:10px;box-shadow:0 1px 5px #d7deea}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}} th,td{{padding:8px 10px;border-bottom:1px solid #e2e7ef;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} code{{background:#edf1f7;padding:2px 5px}}
</style></head><body>
<h1>Predictive experiment: {escape(loaded.experiment_id)}</h1>
<p class="outcome"><strong>Outcome: <code>{escape(conclusion_code)}</code>.</strong> {escape(conclusion_text)}</p>
<div class="card"><h2>What ran</h2><p>Frozen as-of {loaded.as_of_date.date().isoformat()}; assets {escape(', '.join(loaded.statistical.assets))}; next-session {loaded.statistical.horizon_sessions}-session target; weekly W-FRI decisions.</p><p>Primary selected <code>{escape(primary.selected_model)}</code>, then that family was locked into the raw-stitch sensitivity. Its independent raw-only development preference was <code>{escape(str(raw_sensitivity.development_selected_model))}</code> and remained diagnostic. Family selection used only whole folds ending before the global earliest-holdout cutoff. Portfolio evidence uses the common all-asset holdout intersection.</p></div>
<div class="card"><h2>Primary delivery-anchor-neutralized metrics</h2>{primary_table}</div>
<div class="card"><h2>Raw vendor-stitch sensitivity</h2>{raw_table}</div>
<div class="card"><h2>Selected-forecast cost paths</h2>{cost_table}<p>Close-only execution is conservative: the next-session close changes the position and P&amp;L begins on the following interval. A 3 bps roll charge applies to absolute held exposure on every preregistered delivery-month anchor session; it is not observed two-leg roll P&amp;L.</p></div>
<div class="card"><h2>Point-in-time controls</h2><ul><li>USDA CSV releases activate on the first market session strictly after next-day CSV availability; revisions last only through that week's decision, while stocks-to-use carries with age.</li><li>NOAA observations are lagged three calendar days and backward-asof joined.</li><li>All preprocessing, alpha selection, purging, and forecast scaling use training information only.</li></ul></div>
<div class="card"><h2>Data lineage</h2>{lineage_table}</div>
<div class="card"><h2>Claim boundary</h2><p>This is an exploratory falsification test, not a deployable model. Vendor continuous futures, endpoint weather histories, and overlapping labels prevent a validated predictive or tradability claim. Confirmation requires individual contracts, causal roll P&amp;L, dependence-aware uncertainty, and a newly untouched holdout.</p></div>
</body></html>"""


def _write_result_tables(
    output_dir: Path, variant: str, result: PredictiveExperimentResult
) -> list[Path]:
    root = output_dir / variant
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        "predictions.csv.gz": result.predictions,
        "metrics_overall.csv": result.metrics_overall,
        "metrics_by_asset.csv": result.metrics_by_asset,
        "metrics_by_crop_year.csv": result.metrics_by_crop_year,
        "folds.csv": result.folds,
        "coefficients.csv.gz": result.coefficients,
        "hyperparameters.csv.gz": result.hyperparameters,
        "weekly_panel.csv.gz": result.weekly_panel.reset_index(),
    }
    paths: list[Path] = []
    for name, frame in frames.items():
        path = root / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        paths.append(path)
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_model": result.selected_model,
                "development_selected_model": result.development_selected_model,
                "holdout_crop_years": {
                    asset: list(years) for asset, years in result.holdout_crop_years.items()
                },
                "feature_groups": {
                    name: list(columns) for name, columns in result.feature_groups.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(selection)
    return paths


def _manifest(output_dir: Path, paths: Iterable[Path]) -> Path:
    entries = []
    for path in sorted(set(paths)):
        entries.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    target = output_dir / "manifest.json"
    target.write_text(
        json.dumps({"files": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def execute_predictive_experiment(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    refresh_data: bool = False,
    bundle: PredictiveDataBundle | None = None,
) -> PredictiveRunArtifacts:
    """Acquire inputs and execute all preregistered variants exactly once."""

    loaded = load_predictive_config(config_path)
    output = Path(output_dir).resolve()
    state_path = output / "experiment_state.json"
    if state_path.exists():
        raise RuntimeError(
            f"experiment output already has a holdout ledger: {state_path}; "
            "use a new experiment ID/output directory instead of retuning this holdout"
        )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"experiment output directory is not empty: {output}; use a fresh directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        bundle = acquire_predictive_data(
            data_dir,
            start=loaded.start_date,
            end=loaded.as_of_date,
            refresh=refresh_data,
        )
    prices = bundle.market_prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[
        (prices.index >= loaded.start_date) & (prices.index <= loaded.as_of_date),
        list(loaded.statistical.assets),
    ]
    usda_aligned = align_usda_to_market(
        bundle.usda_features,
        prices,
        assets=loaded.statistical.assets,
        weekly_rule=loaded.statistical.weekly_rule,
    )
    weather_aligned = align_weather_to_market(
        bundle.weather_features,
        prices,
        assets=loaded.statistical.assets,
        publication_lag_days=3,
    )

    # Resolve the exact holdout calendar before opening it. This creates no
    # forecasts and inspects no outcome metric; it binds the subsequent run to
    # the dated input/code snapshot recorded in the state ledger.
    preopen_daily, _ = build_daily_prediction_panel(
        prices,
        usda_features=usda_aligned,
        weather_features=weather_aligned,
        config=loaded.statistical,
        proxy_input=True,
    )
    preopen_weekly = sample_weekly_panel(preopen_daily, loaded.statistical)
    planned_holdout_years = {
        asset: [
            int(fold.test_crop_year)
            for fold in make_crop_year_folds(
                preopen_weekly, asset, loaded.statistical
            )
            if fold.is_holdout
        ]
        for asset in loaded.statistical.assets
    }
    preopen_folds = tuple(
        fold
        for asset in loaded.statistical.assets
        for fold in make_crop_year_folds(
            preopen_weekly, asset, loaded.statistical
        )
    )
    selection_cutoff, selection_folds = development_selection_plan(
        preopen_weekly, preopen_folds, loaded.statistical
    )

    state = {
        "experiment_id": loaded.experiment_id,
        "config_sha256": loaded.sha256,
        "status": "holdout_opened_for_predeclared_variants",
        "variants": [
            "delivery_anchor_neutralized_primary",
            "raw_vendor_stitch_sensitivity",
        ],
        "planned_holdout_crop_years": planned_holdout_years,
        "global_selection_cutoff": selection_cutoff.date().isoformat(),
        "development_selection_folds": [
            {"asset": asset, "crop_year": year}
            for asset, year in selection_folds
        ],
        "input_hashes": {
            "market_prices": _dataframe_sha256(prices),
            "usda_point_in_time": _dataframe_sha256(usda_aligned),
            "weather_point_in_time": _dataframe_sha256(weather_aligned),
        },
        "implementation_hashes": {
            "predictive_data.py": _file_sha256(Path(__file__).with_name("predictive_data.py")),
            "predictive_experiment.py": _file_sha256(
                Path(__file__).with_name("predictive_experiment.py")
            ),
            "predictive_runner.py": _file_sha256(Path(__file__)),
        },
        "tuning_after_open": False,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # These are the only two model executions. Both treatments and all choices
    # were committed in the YAML before this state ledger was written.
    primary = run_predictive_experiment(
        prices,
        usda_features=usda_aligned,
        weather_features=weather_aligned,
        config=loaded.statistical,
        proxy_input=True,
    )
    raw_sensitivity = run_predictive_experiment(
        prices,
        usda_features=usda_aligned,
        weather_features=weather_aligned,
        config=replace(loaded.statistical, neutralize_proxy_roll_gaps=False),
        proxy_input=True,
        holdout_model_override=primary.selected_model,
    )
    primary_daily, primary_costs = build_costed_portfolio(
        primary,
        loaded.portfolio,
        variant="delivery_anchor_neutralized_primary",
        split="holdout",
    )
    raw_daily, raw_costs = build_costed_portfolio(
        raw_sensitivity,
        loaded.portfolio,
        variant="raw_vendor_stitch_sensitivity",
        split="holdout",
    )
    portfolio_daily = pd.concat([primary_daily, raw_daily], ignore_index=True)
    portfolio_metrics = pd.concat([primary_costs, raw_costs], ignore_index=True)
    conclusion_code, conclusion_text = _experiment_conclusion(
        primary, raw_sensitivity, portfolio_metrics
    )

    artifact_paths: list[Path] = [state_path]
    frozen_config = output / "preregistered_config.yaml"
    frozen_config.write_text(loaded.path.read_text(encoding="utf-8"), encoding="utf-8")
    artifact_paths.append(frozen_config)
    aligned_dir = output / "aligned_inputs"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    usda_path = aligned_dir / "usda_point_in_time.csv.gz"
    weather_path = aligned_dir / "weather_point_in_time.csv.gz"
    lineage_path = output / "data_lineage.csv"
    usda_aligned.to_csv(usda_path, index=False, compression="gzip")
    weather_aligned.to_csv(weather_path, index=False, compression="gzip")
    bundle.lineage.to_csv(lineage_path, index=False)
    artifact_paths.extend([usda_path, weather_path, lineage_path])
    artifact_paths.extend(_write_result_tables(output, "primary", primary))
    artifact_paths.extend(_write_result_tables(output, "raw_sensitivity", raw_sensitivity))
    daily_path = output / "portfolio_daily_cost_sensitivities.csv.gz"
    portfolio_path = output / "portfolio_metrics.csv"
    portfolio_daily.to_csv(daily_path, index=False, compression="gzip")
    portfolio_metrics.to_csv(portfolio_path, index=False)
    artifact_paths.extend([daily_path, portfolio_path])

    markdown = _markdown_report(
        loaded,
        bundle,
        primary,
        raw_sensitivity,
        portfolio_metrics,
        conclusion_code,
        conclusion_text,
    )
    html = _html_report(
        loaded,
        bundle,
        primary,
        raw_sensitivity,
        portfolio_metrics,
        conclusion_code,
        conclusion_text,
    )
    report_markdown = output / "report.md"
    report_html = output / "report.html"
    report_markdown.write_text(markdown, encoding="utf-8")
    report_html.write_text(html, encoding="utf-8")
    artifact_paths.extend([report_markdown, report_html])
    result_json = output / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "experiment_id": loaded.experiment_id,
                "status": "completed",
                "conclusion": conclusion_code,
                "conclusion_text": conclusion_text,
                "primary_selected_model": primary.selected_model,
                "raw_sensitivity_selected_model": raw_sensitivity.selected_model,
                "report": str(report_html),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(result_json)
    state["status"] = "completed_no_post_holdout_tuning"
    state["conclusion"] = conclusion_code
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _manifest(output, artifact_paths)
    return PredictiveRunArtifacts(
        output_dir=output,
        primary=primary,
        raw_sensitivity=raw_sensitivity,
        portfolio_daily=portfolio_daily,
        portfolio_metrics=portfolio_metrics,
        conclusion=conclusion_code,
        report_markdown=report_markdown,
        report_html=report_html,
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agri-macro-predictive-experiment",
        description="Run the frozen real-data ZW/ZC/ZS predictive experiment",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/predictive_experiment.yaml"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/predictive_experiment_v1")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh-data", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        loaded = load_predictive_config(args.experiment_config)
        output = args.output_dir or Path("reports") / (
            f"{loaded.experiment_id}_{loaded.as_of_date.strftime('%Y%m%d')}"
        )
        result = execute_predictive_experiment(
            config_path=args.experiment_config,
            data_dir=args.data_dir,
            output_dir=output,
            refresh_data=bool(args.refresh_data),
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": result.conclusion,
                    "output_dir": str(result.output_dir),
                    "report": str(result.report_html),
                    "primary_selected_model": result.primary.selected_model,
                    "raw_sensitivity_selected_model": result.raw_sensitivity.selected_model,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"agri-macro predictive-experiment: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "LoadedPredictiveConfig",
    "PortfolioCostConfig",
    "PredictiveRunArtifacts",
    "align_usda_to_market",
    "align_weather_to_market",
    "build_costed_portfolio",
    "execute_predictive_experiment",
    "load_predictive_config",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
