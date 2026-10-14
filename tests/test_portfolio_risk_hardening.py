"""Regression tests for aggregate commodity caps and covariance validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.portfolio import construct_portfolio, expected_portfolio_volatility
from agri_macro_quant.risk import portfolio_risk_report, repair_covariance, risk_contributions


def _unconstrained_except_commodity(limit: float) -> dict[str, float]:
    return {
        "target_volatility": 2.0,
        "max_position_weight": 1.0,
        "max_gross_exposure": 2.0,
        "max_net_exposure": 2.0,
        "max_asset_class_exposure": 1.0,
        "max_commodity_exposure": limit,
        "max_fertilizer_equity_exposure": 1.0,
        "min_signal": 0.0,
    }


def test_max_commodity_exposure_combines_grain_and_energy() -> None:
    signals = pd.Series({"ZW": 1.0, "NG": 1.0, "DXY": 0.8})
    covariance = pd.DataFrame(
        np.eye(3) * 0.04,
        index=signals.index,
        columns=signals.index,
    )
    limit = 0.30

    weights = construct_portfolio(
        signals,
        covariance=covariance,
        config=_unconstrained_except_commodity(limit),
    )

    assert set(weights.index) == set(signals.index)
    assert weights["ZW"] > 0.0 and weights["NG"] > 0.0
    assert weights.loc[["ZW", "NG"]].abs().sum() <= limit + 1e-12
    assert weights.attrs["commodity_gross_exposure"] <= limit + 1e-12


def test_custom_commodity_asset_classes_do_not_invent_positions() -> None:
    signals = pd.Series({"ZW": 1.0, "CF": 1.0})
    covariance = pd.DataFrame(
        np.eye(2) * 0.04,
        index=signals.index,
        columns=signals.index,
    )
    config = _unconstrained_except_commodity(0.20)
    config["commodity_asset_classes"] = ("grain", "fertilizer_equity")

    weights = construct_portfolio(signals, covariance=covariance, config=config)

    assert set(weights.index) == {"ZW", "CF"}
    assert weights.abs().sum() <= 0.20 + 1e-12


def test_indefinite_covariance_is_repaired_without_erasing_risk() -> None:
    labels = ["ZW", "NG"]
    invalid = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=labels, columns=labels)
    weights = pd.Series([1.0, -1.0], index=labels)

    repaired = repair_covariance(invalid)
    diagnostics = repaired.attrs["repair_diagnostics"]

    assert diagnostics["repaired"] is True
    assert diagnostics["eigenvalues_clipped"] == 1
    assert diagnostics["min_correlation_eigenvalue_before"] < 0.0
    assert diagnostics["min_covariance_eigenvalue_after"] > 0.0
    np.testing.assert_allclose(np.diag(repaired), [1.0, 1.0], rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(repaired.to_numpy()).min() > 0.0

    expected_volatility = expected_portfolio_volatility(weights, invalid)
    contributions = risk_contributions(weights, invalid)
    report = portfolio_risk_report(weights, covariance=invalid)
    assert expected_volatility > 0.0
    assert contributions.attrs["expected_volatility"] > 0.0
    assert contributions.attrs["covariance_repair"]["repaired"] is True
    assert report["expected_volatility"] > 0.0
    assert report["covariance_repair"]["repaired"] is True


def test_ordinary_psd_covariance_is_preserved_and_reported_clean() -> None:
    labels = ["ZW", "NG"]
    ordinary = pd.DataFrame([[1.0, 0.2], [0.2, 0.5]], index=labels, columns=labels)

    repaired = repair_covariance(ordinary)

    pd.testing.assert_frame_equal(repaired, ordinary)
    diagnostics = repaired.attrs["repair_diagnostics"]
    assert diagnostics["repaired"] is False
    assert diagnostics["eigenvalues_clipped"] == 0
    assert diagnostics["min_covariance_eigenvalue_after"] > 0.0


def test_portfolio_exposes_covariance_repair_diagnostics() -> None:
    labels = ["ZW", "NG"]
    signals = pd.Series([1.0, -1.0], index=labels)
    invalid = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=labels, columns=labels)
    config = _unconstrained_except_commodity(2.0)

    weights = construct_portfolio(signals, covariance=invalid, config=config)

    diagnostics = weights.attrs["covariance_repair"]
    assert diagnostics["repaired"] is True
    assert diagnostics["eigenvalues_clipped"] == 1
    assert weights.attrs["expected_volatility"] > 0.0


def test_unpriced_signal_cannot_receive_fallback_volatility_position() -> None:
    dates = pd.bdate_range("2024-01-02", periods=60, name="date")
    signals = pd.DataFrame(
        {
            "date": [dates[-1]],
            "asset": ["CF"],
            "signal": [1.0],
            "confidence": [100.0],
        }
    )
    returns = pd.DataFrame({"CF": np.nan}, index=dates)

    weights = construct_portfolio(signals, returns=returns)

    assert weights["CF"] == 0.0
    assert "CF" in weights.attrs["tradability_exclusions"]
