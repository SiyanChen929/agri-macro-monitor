"""Adversarial tests for causal portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.backtest import (
    compare_benchmarks,
    run_backtest,
    slice_backtest_result,
)
from agri_macro_quant.pipeline import _traded_returns


NO_FRICTION = {
    "execution_lag": 0,
    "transaction_cost_bps": 0.0,
    "slippage_bps": 0.0,
    "commission_bps": 0.0,
    "futures_roll_cost_bps": 0.0,
    "amortize_roll_cost": False,
    "apply_drawdown_control": False,
}


def test_constant_daily_target_rebalances_from_post_return_drift() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3, name="date")
    returns = pd.DataFrame(
        {"A": [0.10, 0.00, 0.00], "B": [0.00, 0.00, 0.00]},
        index=dates,
    )
    target = pd.DataFrame(0.50, index=dates, columns=["A", "B"])
    result = run_backtest(
        returns=returns,
        target_weights=target,
        config={
            **NO_FRICTION,
            "transaction_cost_bps": 10.0,
        },
    )

    first_day_net_growth = 1.0 + 0.05 - 1.0 * 10.0 / 10_000.0
    drifted_a = 0.50 * 1.10 / first_day_net_growth
    drifted_b = 0.50 / first_day_net_growth
    expected_second_day_turnover = abs(0.50 - drifted_a) + abs(0.50 - drifted_b)

    assert np.isclose(result.turnover.iloc[0], 1.0)
    assert np.isclose(result.turnover.iloc[1], expected_second_day_turnover)
    assert result.turnover.iloc[1] > 0.0
    assert np.isclose(
        result.costs["transaction_cost"].iloc[1],
        expected_second_day_turnover * 10.0 / 10_000.0,
    )


def test_target_is_not_implicitly_rebalanced_without_a_new_event() -> None:
    dates = pd.bdate_range("2024-02-01", periods=2, name="date")
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.00]}, index=dates)
    one_time_target = pd.DataFrame(
        {"A": [0.50], "B": [0.50]},
        index=dates[:1],
    )

    result = run_backtest(
        returns=returns,
        target_weights=one_time_target,
        config=NO_FRICTION,
    )

    assert np.isclose(result.positions.loc[dates[1], "A"], 0.55 / 1.05)
    assert np.isclose(result.positions.loc[dates[1], "B"], 0.50 / 1.05)
    assert result.turnover.iloc[1] == 0.0


def test_nan_target_leg_means_hold_while_explicit_zero_means_flatten() -> None:
    dates = pd.bdate_range("2024-02-05", periods=3, name="date")
    returns = pd.DataFrame(
        {"A": [0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
        index=dates,
    )
    target = pd.DataFrame(
        {"A": [0.5, np.nan, 0.0], "B": [0.5, 0.5, np.nan]},
        index=dates,
    )

    result = run_backtest(
        returns=returns,
        target_weights=target,
        config=NO_FRICTION,
    )

    assert result.positions.loc[dates[0]].to_dict() == {"A": 0.5, "B": 0.5}
    drifted_a = 0.55 / 1.05
    drifted_b = 0.50 / 1.05
    assert np.isclose(result.positions.loc[dates[1], "A"], drifted_a)
    assert result.positions.loc[dates[1], "B"] == 0.5
    assert np.isclose(result.turnover.iloc[1], 0.5 - drifted_b)
    assert result.positions.loc[dates[2]].to_dict() == {"A": 0.0, "B": 0.5}
    assert np.isclose(result.turnover.iloc[2], drifted_a)


def test_sparse_prices_preserve_move_to_next_valid_mark() -> None:
    dates = pd.bdate_range("2024-03-04", periods=3, name="date")
    prices = pd.DataFrame({"A": [100.0, np.nan, 110.0]}, index=dates)
    one_time_target = pd.DataFrame({"A": [1.0]}, index=dates[:1])

    result = run_backtest(
        prices=prices,
        target_weights=one_time_target,
        config=NO_FRICTION,
    )

    expected_gross = pd.Series([0.0, 0.0, 0.10], index=dates, name="gross_return")
    pd.testing.assert_series_equal(
        result.gross_returns,
        expected_gross,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    assert np.isclose(result.equity_curve.iloc[-1], 1.10)
    assert result.metrics["missing_return_observations"] == 2.0
    assert result.metrics["unpriced_exposure_observations"] == 1.0
    assert result.metrics["unpriced_exposure_periods"] == 1.0
    assert result.positions["A"].tolist() == [1.0, 1.0, 1.0]
    assert result.turnover.tolist() == [1.0, 0.0, 0.0]


def test_deferred_new_position_does_not_receive_pre_trade_catchup_return() -> None:
    dates = pd.bdate_range("2024-03-04", periods=3, name="date")
    returns = pd.DataFrame({"A": [np.nan, 0.10, 0.0]}, index=dates)
    # Repeat the target on the repricing date: the fresh instruction updates
    # the desired close target but must not jump ahead of the catch-up return.
    target = pd.DataFrame({"A": [1.0, 1.0]}, index=dates[:2])

    result = run_backtest(
        returns=returns,
        target_weights=target,
        config=NO_FRICTION,
    )

    # No mark exists when the instruction arrives.  The target is filled only
    # after the next valid interval, so the prior +10% is not credited.
    assert result.positions["A"].tolist() == [0.0, 0.0, 1.0]
    assert result.turnover.tolist() == [0.0, 1.0, 0.0]
    assert result.gross_returns.tolist() == [0.0, 0.0, 0.0]
    assert result.equity_curve.iloc[-1] == 1.0


def test_pipeline_traded_returns_preserve_move_across_missing_mark() -> None:
    dates = pd.bdate_range("2024-03-04", periods=3, name="date")
    market = pd.DataFrame(
        {
            # The aligned decision panel carries the prior mark on the middle
            # date; observation metadata identifies it as stale/nontradable.
            "wheat": [100.0, 100.0, 110.0],
            "us_2y": [4.00, np.nan, 4.25],
        },
        index=dates,
    )
    observation_dates = pd.DataFrame(
        {"wheat": [dates[0], dates[0], dates[2]]},
        index=dates,
    )

    returns = _traded_returns(market, observation_dates)

    assert pd.isna(returns.loc[dates[0], "ZW"])
    assert pd.isna(returns.loc[dates[1], "ZW"])
    assert np.isclose(returns.loc[dates[2], "ZW"], 0.10)
    # A cash-yield change is not an investable total return.  US2Y remains a
    # regime/macro input and is intentionally excluded from portfolio P&L.
    assert "US2Y" not in returns


def test_rebalance_frequency_changes_executable_event_cadence() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10, name="date")
    returns = pd.DataFrame({"A": 0.0}, index=dates)
    target = pd.DataFrame(
        {"A": np.where(np.arange(len(dates)) % 2 == 0, 0.5, -0.5)},
        index=dates,
    )

    daily = run_backtest(
        returns=returns,
        target_weights=target,
        config={**NO_FRICTION, "rebalance_frequency": "daily"},
    )
    weekly = run_backtest(
        returns=returns,
        target_weights=target,
        config={**NO_FRICTION, "rebalance_frequency": "weekly"},
    )

    assert int(daily.turnover.gt(0.0).sum()) > int(weekly.turnover.gt(0.0).sum())


def test_monthly_rebalance_preserves_actual_decision_date_before_lag() -> None:
    dates = pd.DatetimeIndex(
        ["2021-01-28", "2021-01-29", "2021-02-01", "2021-02-02"],
        name="date",
    )
    returns = pd.DataFrame({"A": 0.0}, index=dates)
    target = pd.DataFrame({"A": [1.0]}, index=dates[[1]])

    result = run_backtest(
        returns=returns,
        target_weights=target,
        config={
            **NO_FRICTION,
            "execution_lag": 1,
            "rebalance_frequency": "monthly",
        },
    )

    assert result.positions.loc[pd.Timestamp("2021-01-29"), "A"] == 0.0
    assert result.positions.loc[pd.Timestamp("2021-02-01"), "A"] == 1.0


def test_missing_direct_return_is_unpriced_and_disclosed() -> None:
    dates = pd.bdate_range("2024-04-01", periods=3, name="date")
    returns = pd.DataFrame({"A": [0.05, np.nan, 0.02]}, index=dates)
    one_time_target = pd.DataFrame({"A": [1.0]}, index=dates[:1])

    result = run_backtest(
        returns=returns,
        target_weights=one_time_target,
        config=NO_FRICTION,
    )

    assert result.gross_returns.iloc[1] == 0.0
    assert result.metrics["missing_return_observations"] == 1.0
    assert result.metrics["unpriced_exposure_observations"] == 1.0
    assert result.metrics["unpriced_exposure_periods"] == 1.0


def test_two_ten_percent_log_returns_match_simple_returns() -> None:
    dates = pd.bdate_range("2024-05-01", periods=2, name="date")
    simple_returns = pd.DataFrame({"A": [0.10, 0.10]}, index=dates)
    log_returns = np.log1p(simple_returns)
    one_time_target = pd.DataFrame({"A": [1.0]}, index=dates[:1])

    simple = run_backtest(
        returns=simple_returns,
        target_weights=one_time_target,
        config={**NO_FRICTION, "return_method": "simple"},
    )
    logarithmic = run_backtest(
        returns=log_returns,
        target_weights=one_time_target,
        config={**NO_FRICTION, "return_method": "log"},
    )

    pd.testing.assert_series_equal(
        logarithmic.returns,
        simple.returns,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    pd.testing.assert_series_equal(
        logarithmic.equity_curve,
        simple.equity_curve,
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    assert np.isclose(logarithmic.equity_curve.iloc[-1], 1.21)


def test_evaluation_slice_retains_pre_start_holdings_and_benchmark_state() -> None:
    dates = pd.bdate_range("2024-01-01", periods=9, name="date")
    returns = pd.DataFrame(
        {"A": [0.0, 0.10, 0.0, 0.0, 0.0, 0.02, 0.01, 0.0, 0.0]},
        index=dates,
    )
    target = pd.DataFrame({"A": [0.5]}, index=dates[:1])
    config = {
        **NO_FRICTION,
        "execution_lag": 1,
        "rebalance_frequency": "weekly",
        "transaction_cost_bps": 100.0,
    }
    full = run_backtest(returns=returns, target_weights=target, config=config)
    evaluation_start = dates[7]
    sliced = slice_backtest_result(
        full,
        evaluation_start,
        config=config,
        asset_returns=returns,
    )

    assert sliced.positions.iloc[0, 0] == full.positions.loc[evaluation_start, "A"]
    assert sliced.turnover.iloc[0] == 0.0
    assert sliced.costs["total_cost"].iloc[0] == 0.0

    comparison = compare_benchmarks(
        sliced,
        returns,
        config={**config, "evaluation_start": evaluation_start},
    )
    # Equal-weight and static benchmarks were already established before the
    # evaluation boundary; slicing must not charge a fabricated opening trade.
    assert comparison.loc["equal_weight", "total_transaction_cost"] == 0.0
    assert comparison.loc["static_long_A", "total_transaction_cost"] == 0.0
