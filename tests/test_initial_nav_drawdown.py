"""Regression tests for drawdowns measured from the pre-return NAV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.backtest import performance_metrics, run_backtest
from agri_macro_quant.risk import drawdown_series, portfolio_risk_report
from agri_macro_quant.scenarios import historical_stress_test


NO_FRICTION = {
    "execution_lag": 0,
    "transaction_cost_bps": 0.0,
    "slippage_bps": 0.0,
    "commission_bps": 0.0,
    "futures_roll_cost_bps": 0.0,
    "amortize_roll_cost": False,
    "apply_drawdown_control": False,
}


def test_return_drawdown_includes_initial_nav_and_preserves_index() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3, name="date")
    returns = pd.Series([-0.10, 0.20, -0.05], index=dates)

    drawdown = drawdown_series(returns, "returns")

    expected = pd.Series([-0.10, 0.0, -0.05], index=dates, name="drawdown")
    pd.testing.assert_series_equal(drawdown, expected)


def test_equity_drawdown_uses_recorded_non_unit_initial_capital() -> None:
    dates = pd.bdate_range("2024-02-01", periods=3, name="date")
    equity = pd.Series([90.0, 108.0, 102.6], index=dates)
    equity.attrs["initial_value"] = 100.0

    drawdown = drawdown_series(equity, "equity")

    expected = pd.Series([-0.10, 0.0, -0.05], index=dates, name="drawdown")
    pd.testing.assert_series_equal(drawdown, expected)


def test_backtest_and_metrics_capture_first_row_loss() -> None:
    dates = pd.bdate_range("2024-03-01", periods=2, name="date")
    returns = pd.DataFrame({"A": [-0.10, 0.05]}, index=dates)
    target = pd.DataFrame({"A": [1.0]}, index=dates[:1])

    result = run_backtest(
        returns=returns,
        target_weights=target,
        config={**NO_FRICTION, "initial_capital": 100.0},
    )

    assert np.isclose(result.equity_curve.iloc[0], 90.0)
    assert np.isclose(result.drawdown.iloc[0], -0.10)
    assert np.isclose(result.metrics["maximum_drawdown"], -0.10)
    direct_metrics = performance_metrics(pd.Series([-0.10, 0.05]), config={"annualization": 252})
    assert np.isclose(direct_metrics["maximum_drawdown"], -0.10)


def test_static_replay_and_historical_stress_capture_first_row_loss() -> None:
    dates = pd.bdate_range("2024-04-01", periods=3, name="date")
    returns = pd.DataFrame({"A": [-0.10, 0.20, -0.05]}, index=dates)
    weights = pd.Series({"A": 1.0})

    risk = portfolio_risk_report(weights, returns=returns)
    assert np.isclose(risk["static_weight_replay_drawdown"], -0.05)

    stress = historical_stress_test(
        weights,
        returns,
        periods={"test_window": (dates[0], dates[-1])},
    )
    assert np.isclose(stress.loc["test_window", "maximum_drawdown"], -0.10)


def test_single_row_static_replay_reports_initial_loss() -> None:
    date = pd.Timestamp("2024-05-01")
    returns = pd.DataFrame({"A": [-0.10]}, index=pd.DatetimeIndex([date]))
    weights = pd.Series({"A": 1.0})
    covariance = pd.DataFrame({"A": [0.04]}, index=["A"])

    risk = portfolio_risk_report(weights, returns=returns, covariance=covariance)
    stress = historical_stress_test(
        weights,
        returns,
        periods={"one_day": (date, date)},
    )

    assert np.isclose(risk["static_weight_replay_drawdown"], -0.10)
    assert np.isclose(stress.loc["one_day", "maximum_drawdown"], -0.10)


def test_risk_and_stress_exclude_unpriced_active_weight_days() -> None:
    dates = pd.bdate_range("2024-06-03", periods=3, name="date")
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01], "B": [0.01, np.nan, -0.02]},
        index=dates,
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    risk = portfolio_risk_report(weights, returns=returns)
    stress = historical_stress_test(
        weights,
        returns,
        periods={"window": (dates[0], dates[-1])},
    )

    assert risk["historical_unpriced_days_excluded"] == 1
    assert risk["historical_risk_observations"] == 2
    assert np.isclose(risk["historical_priced_weight_coverage_min"], 0.5)
    assert stress.loc["window", "excluded_unpriced_days"] == 1
    assert stress.loc["window", "priced_observations"] == 2


def test_zero_weight_missing_asset_does_not_delete_stress_observations() -> None:
    dates = pd.bdate_range("2024-07-01", periods=3, name="date")
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03], "B": [np.nan, np.nan, np.nan]},
        index=dates,
    )
    weights = pd.Series({"A": 1.0, "B": 0.0})

    stress = historical_stress_test(
        weights,
        returns,
        periods={"window": (dates[0], dates[-1])},
    )

    assert stress.loc["window", "priced_observations"] == 3
    assert stress.loc["window", "excluded_unpriced_days"] == 0
    assert np.isclose(stress.loc["window", "portfolio_pnl"], 1.01 * 0.98 * 1.03 - 1.0)
