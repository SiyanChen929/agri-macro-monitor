from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from agri_macro_quant.research.predictive_experiment import PredictiveExperimentResult
from agri_macro_quant.research.predictive_runner import (
    PortfolioCostConfig,
    align_usda_to_market,
    align_weather_to_market,
    build_costed_portfolio,
    load_predictive_config,
)


def _market_panel(start: str = "2024-01-01", periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "ZW": np.linspace(100.0, 104.0, periods),
            "ZC": np.linspace(110.0, 115.0, periods),
            "ZS": np.linspace(120.0, 126.0, periods),
        },
        index=dates,
    )


def test_usda_alignment_is_strict_and_revision_is_release_week_only() -> None:
    prices = _market_panel("2024-01-01", 20)
    usda = pd.DataFrame(
        {
            "available_date": [pd.Timestamp("2024-01-09")],
            "release_date": [pd.Timestamp("2024-01-08")],
            "asset": ["ZC"],
            "us_yield_revision": [0.12],
            "us_stocks_to_use": [0.085],
        }
    )
    aligned = align_usda_to_market(usda, prices, assets=("ZC",))
    corn = aligned.set_index("date")

    # Jan 9 is the CSV availability date; strict next-session activation is Jan 10.
    assert corn.loc[pd.Timestamp("2024-01-09"), "us_yield_revision"] == 0.0
    assert pd.isna(corn.loc[pd.Timestamp("2024-01-09"), "us_stocks_to_use"])
    assert corn.loc[pd.Timestamp("2024-01-10"), "us_yield_revision"] == pytest.approx(0.12)
    assert corn.loc[pd.Timestamp("2024-01-12"), "us_yield_revision"] == pytest.approx(0.12)
    assert corn.loc[pd.Timestamp("2024-01-15"), "us_yield_revision"] == 0.0
    assert corn.loc[pd.Timestamp("2024-01-15"), "us_stocks_to_use"] == pytest.approx(0.085)
    assert corn.loc[pd.Timestamp("2024-01-15"), "days_since_usda_release"] == 5.0


def test_weather_alignment_reconstructs_three_day_availability() -> None:
    prices = _market_panel("2024-01-01", 10)
    weather = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2024-01-02")],
            # This deliberately incorrect field must be ignored by the runner.
            "available_date": [pd.Timestamp("2024-01-02")],
            "asset": ["ZW"],
            "temperature_anomaly_14d": [2.5],
        }
    )
    aligned = align_weather_to_market(weather, prices, assets=("ZW",))
    wheat = aligned.set_index("date")

    assert pd.isna(wheat.loc[pd.Timestamp("2024-01-04"), "temperature_anomaly_14d"])
    assert wheat.loc[pd.Timestamp("2024-01-05"), "temperature_anomaly_14d"] == pytest.approx(2.5)


def _portfolio_fixture() -> PredictiveExperimentResult:
    dates = pd.bdate_range("2024-01-01", periods=24)
    daily = pd.DataFrame(
        {
            "date": dates,
            "asset": "ZW",
            "price": 100.0 * np.cumprod(np.full(len(dates), 1.001)),
            "proxy_roll_gap_flag": [position == 12 for position in range(len(dates))],
        }
    ).set_index(["date", "asset"])
    weekly_dates = pd.DatetimeIndex([dates[0], dates[5], dates[10], dates[15]])
    weekly = pd.DataFrame(
        {
            "date": weekly_dates,
            "asset": "ZW",
            "crop_year": 2023,
            "target_return_21d": [0.02, 0.01, 0.03, 0.015],
            "target_end_date": weekly_dates + pd.Timedelta(days=2),
        }
    ).set_index(["date", "asset"])
    predictions = pd.DataFrame(
        {
            "date": weekly_dates[1:],
            "asset": "ZW",
            "crop_year": 2024,
            "split": "holdout",
            "model": "ridge_core",
            "actual": 0.01,
            "prediction": [0.02, -0.01, 0.015],
            "target_end_date": weekly_dates[1:] + pd.Timedelta(days=21),
        }
    )
    folds = pd.DataFrame(
        {
            "asset": ["ZW"],
            "test_crop_year": [2024],
            "split": ["holdout"],
            "train_crop_years": ["2023"],
            "purge_cutoff": [pd.Timestamp("2024-12-01")],
        }
    )
    from agri_macro_quant.research.predictive_experiment import PredictiveExperimentConfig

    return PredictiveExperimentResult(
        config=PredictiveExperimentConfig(assets=("ZW",)),
        proxy_input=True,
        selected_model="ridge_core",
        feature_groups={"core": (), "usda": (), "weather": ()},
        holdout_crop_years={"ZW": (2024,)},
        daily_panel=daily,
        weekly_panel=weekly,
        folds=folds,
        predictions=predictions,
        coefficients=pd.DataFrame(),
        hyperparameters=pd.DataFrame(),
        metrics_overall=pd.DataFrame(),
        metrics_by_asset=pd.DataFrame(),
        metrics_by_crop_year=pd.DataFrame(),
    )


def test_cost_path_has_execution_lag_and_monotone_cost_stress() -> None:
    costs = PortfolioCostConfig(
        maximum_asset_weight=1.0 / 3.0,
        maximum_gross_exposure=1.0,
        transaction_cost_bps_one_way=5.0,
        slippage_bps_one_way=2.0,
        commission_bps_one_way=0.0,
        futures_roll_cost_bps=3.0,
        cost_sensitivity_multipliers=(1.0, 1.5, 2.0),
    )
    daily, metrics = build_costed_portfolio(
        _portfolio_fixture(), costs, variant="test"
    )

    one = daily.loc[daily["cost_multiplier"].eq(1.0)].reset_index(drop=True)
    assert one["evaluation_split"].eq("holdout").all()
    # First forecast is at dates[5]. Its first eligible target close is dates[6],
    # and close-to-close P&L does not begin until dates[7].
    assert one.loc[
        one["date"].le(pd.Timestamp("2024-01-09")), "gross_exposure"
    ].eq(0.0).all()
    assert one.loc[
        one["date"].eq(pd.Timestamp("2024-01-10")), "gross_exposure"
    ].iloc[0] > 0.0
    ordered = metrics.sort_values("cost_multiplier")
    assert ordered["terminal_net_nav"].is_monotonic_decreasing
    assert ordered["total_transaction_cost"].is_monotonic_increasing
    assert ordered["total_roll_cost"].is_monotonic_increasing
    assert ordered["total_roll_cost"].iloc[0] > 0.0


def test_frozen_config_loads() -> None:
    loaded = load_predictive_config("config/predictive_experiment.yaml")
    assert loaded.experiment_id == "zw_zc_zs_21d_v1"
    assert loaded.statistical.horizon_sessions == 21
    assert loaded.statistical.minimum_weekly_observations_per_crop_year == 40
    assert loaded.portfolio.cost_sensitivity_multipliers == (1.0, 1.5, 2.0)
    assert loaded.statistical.proxy_roll_anchor_day == 15


def test_sharpe_uses_arithmetic_mean_and_cost_aware_policy_keeps_execution_clock():
    costs = PortfolioCostConfig(1/3, 1.0, 5.0, 2.0, 0.0, 3.0, (1.0, 2.0))
    daily, metrics = build_costed_portfolio(
        _portfolio_fixture(), costs, variant="cost_band", target_policy="cost_aware_band"
    )
    one = daily[daily.cost_multiplier.eq(1.0)]
    two = daily[daily.cost_multiplier.eq(2.0)]
    np.testing.assert_allclose(one.gross_return, two.gross_return)
    np.testing.assert_allclose(one.turnover, two.turnover)
    assert one[one.date.le(pd.Timestamp("2024-01-09"))].gross_return.eq(0).all()
    expected = one.net_return.mean() / one.net_return.std(ddof=0) * np.sqrt(252)
    assert metrics.iloc[0].net_sharpe_zero_rate == pytest.approx(expected)
    assert metrics.iloc[0].net_sharpe_zero_rate != pytest.approx(
        metrics.iloc[0].geometric_return_to_volatility, abs=1e-5
    )


def test_frozen_config_rejects_feature_registry_drift(tmp_path) -> None:
    source = Path("config/predictive_experiment.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["feature_families"]["core"].append("unregistered_feature")
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_families"):
        load_predictive_config(altered)
