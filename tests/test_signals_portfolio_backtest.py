from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.backtest import compare_benchmarks, performance_metrics, run_backtest
from agri_macro_quant.portfolio import aggregate_executable_signals, construct_portfolio
from agri_macro_quant.signals import SIGNAL_COLUMNS, generate_trade_signals


def _signal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    index = pd.bdate_range("2024-01-02", periods=5, name="date")
    factors = pd.DataFrame(
        {
            "weather_stress": [-1.0, -0.2, 0.0, 0.3, 1.2],
            "corn_urea": [0.9, 0.4, 0.0, -0.4, -0.9],
        },
        index=index,
    )
    regimes = pd.DataFrame(
        {
            "weather_shock": [70.0, 65.0, 55.0, 45.0, 35.0],
            "supply_response": [30.0, 35.0, 45.0, 55.0, 65.0],
        },
        index=index,
    )
    transitions = {
        "weather_shock": {"weather_shock": 0.25, "supply_response": 0.75},
        "supply_response": {"weather_shock": 0.20, "supply_response": 0.80},
    }
    return factors, regimes, transitions


def test_trade_signal_schema_ranges_directions_and_explanations() -> None:
    factors, regimes, transitions = _signal_inputs()
    specs = {
        "ZW": {
            "factor_weights": {"weather_stress": 1.0},
            "regime_weights": {},
            "horizon": "1-2 weeks",
            "driver": "weather stress",
            "invalidation": "weather stress normalizes",
        },
        "CF": {
            "factor_weights": {"corn_urea": 1.0},
            "regime_weights": {},
            "horizon": "1-2 months",
            "driver": "crop/input affordability",
            "invalidation": "affordability reverses",
        },
    }

    signals = generate_trade_signals(
        factors,
        regimes,
        transitions,
        config={"signal_specs": specs, "min_signal": 0.05, "winsorize": 0.0},
    )

    assert list(signals.columns) == SIGNAL_COLUMNS
    assert len(signals) == len(factors) * len(specs)
    assert not signals.duplicated(["date", "asset", "signal_type"]).any()
    assert set(signals["asset"]) == set(specs)
    assert signals["signal"].between(-1.0, 1.0).all()
    assert signals["strength"].between(0.0, 1.0).all()
    assert signals["confidence"].between(0.0, 100.0).all()
    np.testing.assert_allclose(signals["strength"], signals["signal"].abs())
    expected_direction = np.select(
        [signals["signal"] > 0.05, signals["signal"] < -0.05],
        ["long", "short"],
        default="flat",
    )
    np.testing.assert_array_equal(signals["direction"], expected_direction)
    assert signals["economic_driver"].str.len().gt(0).all()
    assert signals["invalidation_condition"].str.len().gt(0).all()
    assert signals["expected_next_regime"].isin({"weather_shock", "supply_response"}).all()


def test_signal_history_is_prefix_invariant() -> None:
    factors, regimes, transitions = _signal_inputs()
    specs = {
        "ZW": {
            "factor_weights": {"weather_stress": 1.0},
            "regime_weights": {"weather_shock": 0.25},
        }
    }
    cutoff = 3
    shocked_factors = factors.copy()
    shocked_regimes = regimes.copy()
    shocked_factors.iloc[cutoff:] = 10_000.0
    shocked_regimes.iloc[cutoff:] = shocked_regimes.iloc[cutoff:].iloc[:, ::-1].to_numpy()

    prefix = generate_trade_signals(
        factors.iloc[:cutoff],
        regimes.iloc[:cutoff],
        transitions,
        {"signal_specs": specs},
    )
    extended = generate_trade_signals(
        shocked_factors,
        shocked_regimes,
        transitions,
        {"signal_specs": specs},
    )
    extended = extended.loc[extended["date"].isin(factors.index[:cutoff])].reset_index(drop=True)

    pd.testing.assert_frame_equal(prefix, extended)


def test_portfolio_obeys_position_group_gross_and_net_constraints() -> None:
    signals = pd.Series(
        {"ZW": 1.0, "ZC": 0.9, "CF": -0.8, "NTR": -0.7, "DXY": 0.6}
    )
    covariance = pd.DataFrame(
        np.eye(len(signals)) * 0.04,
        index=signals.index,
        columns=signals.index,
    )
    config = {
        "target_volatility": 1.0,
        "max_position_weight": 0.18,
        "max_gross_exposure": 0.50,
        "max_net_exposure": 0.10,
        "max_commodity_exposure": 0.22,
        "max_fertilizer_equity_exposure": 0.20,
        "max_asset_class_exposure": 0.30,
        "min_signal": 0.0,
    }

    weights = construct_portfolio(signals, covariance=covariance, config=config)

    assert weights.abs().max() <= config["max_position_weight"] + 1e-12
    assert weights.abs().sum() <= config["max_gross_exposure"] + 1e-12
    assert abs(weights.sum()) <= config["max_net_exposure"] + 1e-12
    assert weights.loc[["ZW", "ZC"]].abs().sum() <= config["max_commodity_exposure"] + 1e-12
    assert (
        weights.loc[["CF", "NTR"]].abs().sum()
        <= config["max_fertilizer_equity_exposure"] + 1e-12
    )
    assert np.isclose(weights.attrs["gross_exposure"], weights.abs().sum(), atol=1e-12)
    assert np.isclose(weights.attrs["net_exposure"], weights.sum(), atol=1e-12)


def test_overlapping_factor_exposure_is_capped_without_inventing_or_flipping_positions() -> None:
    signals = pd.Series({"ZW": 1.0, "ZC": 0.8, "ZS": 0.6})
    covariance = pd.DataFrame(
        np.eye(3) * 0.04,
        index=signals.index,
        columns=signals.index,
    )
    factor_exposures = pd.DataFrame(
        {"physical_scarcity": [1.0, 1.0, 1.0, 1.0]},
        index=["ZW", "ZC", "ZS", "MOS"],
    )
    limit = 0.16

    weights = construct_portfolio(
        signals,
        covariance=covariance,
        factor_exposures=factor_exposures,
        config={
            "target_volatility": 1.0,
            "max_position_weight": 1.0,
            "max_gross_exposure": 1.0,
            "max_net_exposure": 1.0,
            "max_asset_class_exposure": 1.0,
            "max_commodity_exposure": 1.0,
            "factor_limits": {"physical_scarcity": limit},
            "min_signal": 0.0,
        },
    )

    assert set(weights.index) == set(signals.index)
    assert (weights >= -1e-14).all()
    exposure = float(weights.dot(factor_exposures.loc[weights.index, "physical_scarcity"]))
    assert exposure <= limit + 1e-12
    assert weights.attrs["factor_exposure"]["physical_scarcity"] <= limit + 1e-12


def test_negative_beta_relative_value_keeps_signed_executable_hedge() -> None:
    date = pd.Timestamp("2024-06-03")
    pair = pd.DataFrame(
        {
            "date": [date],
            "asset": ["ZW/DXY"],
            "signal": [0.8],
            "confidence": [100.0],
            "long_leg": ["ZW"],
            "short_leg": ["DXY"],
            "hedge_ratio": [-0.5],
        }
    )
    covariance = pd.DataFrame(
        np.eye(2) * 0.04,
        index=["ZW", "DXY"],
        columns=["ZW", "DXY"],
    )
    weights = construct_portfolio(
        pair,
        covariance=covariance,
        config={
            "target_volatility": 0.2,
            "max_position_weight": 1.0,
            "max_gross_exposure": 2.0,
            "max_net_exposure": 2.0,
            "max_asset_class_exposure": 2.0,
            "max_commodity_exposure": 2.0,
            "min_signal": 0.0,
        },
    )

    assert weights["ZW"] > 0
    assert weights["DXY"] > 0  # r_ZW - (-beta)*r_DXY is long both legs


def test_signal_ensemble_aggregates_confidence_weighted_signed_evidence() -> None:
    date = pd.Timestamp("2024-06-03")
    signals = pd.DataFrame(
        {
            "date": [date, date],
            "asset": ["ZW", "ZW"],
            "signal": [1.0, -1.0],
            "confidence": [100.0, 1.0],
        }
    )

    aggregated = aggregate_executable_signals(signals).loc["ZW"]

    assert np.isclose(aggregated["confidence"], 0.505)
    assert np.isclose(
        aggregated["signal"] * aggregated["confidence"],
        (1.0 * 1.0 + -1.0 * 0.01) / 2.0,
    )
    assert aggregated["signal"] > 0.95


def test_sharpe_uses_annualized_arithmetic_excess_return() -> None:
    returns = pd.Series([0.10, -0.05])
    metrics = performance_metrics(
        returns,
        config={"annualization": 2, "risk_free_rate": 0.0},
    )
    expected = (returns.mean() * 2.0) / (returns.std(ddof=1) * np.sqrt(2.0))
    assert np.isclose(metrics["sharpe_ratio"], expected)


def test_internal_benchmarks_apply_execution_costs_and_track_turnover() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80, name="date")
    asset_returns = pd.DataFrame(
        {
            "ZW": np.tile([0.01, -0.004], 40),
            "ZC": np.tile([-0.002, 0.006], 40),
        },
        index=dates,
    )
    zero_target = pd.DataFrame(0.0, index=dates, columns=asset_returns.columns)
    strategy = run_backtest(
        returns=asset_returns,
        target_weights=zero_target,
        config={"apply_drawdown_control": False, "amortize_roll_cost": False},
    )
    comparison = compare_benchmarks(
        strategy,
        asset_returns,
        config={
            "momentum_lookback": 10,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 0.0,
            "commission_bps": 0.0,
            "futures_roll_cost_bps": 0.0,
            "amortize_roll_cost": False,
            "apply_drawdown_control": False,
        },
    )

    for name in ("equal_weight", "static_long_ZW", "time_series_momentum"):
        assert comparison.loc[name, "annualized_turnover"] > 0
        assert comparison.loc[name, "number_of_trades"] > 0
        assert comparison.loc[name, "total_transaction_cost"] > 0
    assert comparison.loc["equal_weight", "average_gross_exposure"] > 0.5


def test_backtest_uses_lagged_weights_and_exact_transaction_cost_arithmetic() -> None:
    dates = pd.bdate_range("2024-05-06", periods=4, name="date")
    returns = pd.DataFrame({"A": [0.10, 0.20, -0.10, 0.05]}, index=dates)
    target = pd.DataFrame({"A": [1.0, 1.0, -0.5, -0.5]}, index=dates)
    config = {
        "execution_lag": 1,
        "transaction_cost_bps": 10.0,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "futures_roll_cost_bps": 0.0,
        "amortize_roll_cost": False,
        "rebalance_frequency": "daily",
        "initial_capital": 100.0,
        "apply_drawdown_control": False,
    }

    result = run_backtest(returns=returns, target_weights=target, config=config)

    expected_position = pd.Series([0.0, 1.0, 1.0, -0.5], index=dates, name="A")
    # A daily target is a daily rebalance instruction.  After the first
    # invested return the holding drifts slightly above 100% because costs
    # reduce NAV; the next target event trades that drift back to 100%.
    drift_after_day_2 = 1.20 / 1.199
    turnover_day_3 = drift_after_day_2 - 1.0
    cost_day_3 = turnover_day_3 * 0.001
    net_day_3 = -0.10 - cost_day_3
    drift_after_day_3 = 0.90 / (1.0 + net_day_3)
    turnover_day_4 = drift_after_day_3 + 0.5
    cost_day_4 = turnover_day_4 * 0.001
    expected_turnover = pd.Series(
        [0.0, 1.0, turnover_day_3, turnover_day_4],
        index=dates,
        name="turnover",
    )
    expected_cost = pd.Series([0.0, 0.001, cost_day_3, cost_day_4], index=dates)
    expected_gross = pd.Series([0.0, 0.20, -0.10, -0.025], index=dates, name="gross_return")
    expected_net = pd.Series(
        [0.0, 0.199, net_day_3, -0.025 - cost_day_4],
        index=dates,
        name="return",
    )
    expected_equity = 100.0 * (1.0 + expected_net).cumprod()
    expected_equity.name = "equity"

    pd.testing.assert_series_equal(result.positions["A"], expected_position)
    pd.testing.assert_series_equal(result.turnover, expected_turnover)
    pd.testing.assert_series_equal(
        result.costs["transaction_cost"], expected_cost, check_names=False
    )
    pd.testing.assert_series_equal(result.costs["total_cost"], expected_cost, check_names=False)
    pd.testing.assert_series_equal(result.gross_returns, expected_gross)
    pd.testing.assert_series_equal(result.returns, expected_net)
    pd.testing.assert_series_equal(
        result.equity_curve, expected_equity, check_exact=False, rtol=1e-14, atol=1e-14
    )
    assert result.metrics["total_transaction_cost"] == expected_cost.sum()
    assert result.gross_returns.iloc[0] == 0.0  # the day-zero 10% return is untradeable
