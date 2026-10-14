from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri_macro_quant.research import (
    ASSOCIATION_DISCLAIMER,
    cross_correlation_scan,
    distributed_lag_regression,
    granger_causality_test,
    run_lead_lag_analysis,
)


def _leading_series(seed: int = 730, periods: int = 500, true_lag: int = 3) -> tuple[pd.Series, pd.Series]:
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=periods, name="date")
    x_values = generator.normal(size=periods)
    noise = generator.normal(scale=0.25, size=periods)
    y_values = 0.85 * pd.Series(x_values).shift(true_lag).fillna(0.0).to_numpy() + noise
    return (
        pd.Series(y_values, index=index, name="crop_condition"),
        pd.Series(x_values, index=index, name="weather_stress"),
    )


def test_cross_correlation_and_distributed_lag_recover_x_leading_y() -> None:
    y, x = _leading_series()

    correlations = cross_correlation_scan(y, x, range(-6, 7))
    strongest = correlations.loc[correlations["correlation"].abs().idxmax()]
    assert int(strongest["lag"]) == 3
    assert strongest["direction"] == "x_leads_y"
    assert strongest["correlation"] > 0.9
    assert not correlations.loc[correlations["lag"] < 0, "causal_design"].any()

    coefficients = distributed_lag_regression(y, x, [1, 2, 3, 4, 5], hac_lags=4)
    predictor_rows = coefficients.loc[coefficients["term_type"] == "predictor"].set_index("lag")
    assert predictor_rows.loc[3, "coefficient"] == pytest.approx(0.85, abs=0.04)
    assert predictor_rows.loc[3, "p_value"] < 1.0e-10
    assert predictor_rows.loc[3, "t_stat"] > 10.0
    assert (coefficients["uses_future_x"] == False).all()  # noqa: E712


def test_regression_design_never_uses_future_x() -> None:
    y, x = _leading_series(periods=80)
    result = distributed_lag_regression(y, x, [0, 1, 3], include_design=True, covariance="classical")
    design = result.attrs["design_matrix"]

    expected_lag_three = x.shift(3).reindex(design.index)
    pd.testing.assert_series_equal(
        design["weather_stress_lag_3"],
        expected_lag_three.rename("weather_stress_lag_3"),
        check_names=True,
    )
    assert design.index.min() == x.index[3]
    assert result.attrs["term_metadata"]["weather_stress_lag_3"] == ("predictor", 3)
    with pytest.raises(ValueError, match="future x"):
        distributed_lag_regression(y, x, [-1, 1])


def test_granger_and_high_level_outputs_are_deterministic_and_structured() -> None:
    y, x = _leading_series(true_lag=2)
    granger = granger_causality_test(y, x, max_lag=4, include_design=True)

    order_two = granger.loc[granger["lag_order"] == 2].iloc[0]
    assert order_two["status"] == "ok"
    assert order_two["p_value"] < 1.0e-10
    assert bool(order_two["reject_null"])
    designs = granger.attrs["designs"][2]["unrestricted"]
    assert "weather_stress_lag_1" in designs
    assert "weather_stress_lag_2" in designs
    assert all("lag_-" not in column for column in designs.columns)

    frame = pd.concat([y, x], axis=1)
    first = run_lead_lag_analysis(
        frame,
        relationships=[("crop_condition", "weather_stress")],
        lags=range(-4, 5),
        max_lag=4,
        transform="levels",
    )
    second = run_lead_lag_analysis(
        frame,
        relationships=[("crop_condition", "weather_stress")],
        lags=range(-4, 5),
        max_lag=4,
        transform="levels",
    )
    pd.testing.assert_frame_equal(first.cross_correlations, second.cross_correlations)
    pd.testing.assert_frame_equal(first.distributed_lag_coefficients, second.distributed_lag_coefficients)
    pd.testing.assert_frame_equal(first.granger_tests, second.granger_tests)
    assert first.metadata.loc[0, "relationship"] == "weather_stress -> crop_condition"
    assert first.disclaimer == ASSOCIATION_DISCLAIMER
    assert "not structural causality" in first.disclaimer
