from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri_macro_quant.factors import MVP_FACTOR_COLUMNS, build_mvp_factors
from agri_macro_quant.features import (
    apply_release_lags,
    build_features,
    expanding_zscore,
    robust_zscore,
    rolling_percentile_rank,
    rolling_winsorize,
    rolling_zscore,
)
from agri_macro_quant.regimes import REGIME_NAMES, score_regimes
from agri_macro_quant.transitions import estimate_transition_probabilities


TRANSFORMS = [
    pytest.param(
        lambda x: rolling_winsorize(
            x, window=20, limits=(0.05, 0.95), min_periods=5, history_lag=1
        ),
        id="rolling-winsorize",
    ),
    pytest.param(
        lambda x: rolling_zscore(
            x, window=20, min_periods=5, history_lag=1, clip=4.0
        ),
        id="rolling-zscore",
    ),
    pytest.param(
        lambda x: robust_zscore(
            x, window=20, min_periods=5, history_lag=1, clip=4.0
        ),
        id="robust-zscore",
    ),
    pytest.param(
        lambda x: expanding_zscore(
            x, min_periods=5, history_lag=1, clip=4.0
        ),
        id="expanding-zscore",
    ),
    pytest.param(
        lambda x: rolling_percentile_rank(
            x, window=20, min_periods=5, history_lag=1
        ),
        id="rolling-percentile",
    ),
]


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_trailing_transforms_are_prefix_invariant(transform) -> None:
    index = pd.bdate_range("2021-01-04", periods=90, name="date")
    base = pd.Series(
        np.sin(np.arange(90) / 5.0) + np.arange(90) * 0.01,
        index=index,
        name="x",
    )
    cutoff = 60
    with_future_shock = base.copy()
    with_future_shock.iloc[cutoff:] = np.linspace(10_000.0, -20_000.0, len(base) - cutoff)

    prefix = transform(base.iloc[:cutoff])
    extended = transform(with_future_shock).iloc[:cutoff]

    pd.testing.assert_series_equal(prefix, extended, check_exact=False, rtol=1e-13, atol=1e-13)


def test_release_lag_and_full_feature_builder_do_not_leak_future_data() -> None:
    index = pd.bdate_range("2021-06-01", periods=75, name="date")
    frame = pd.DataFrame(
        {
            "price": 100.0 + np.arange(75) + 2.0 * np.sin(np.arange(75) / 4.0),
            "release": np.arange(75, dtype=float),
        },
        index=index,
    )
    config = {
        "publication_lags": {"release": 2},
        "changes": [3, 10],
        "zscore_windows": [10, 20],
        "percentile_windows": [20],
        "momentum_windows": [5],
        "volatility_windows": [10],
        "trend_windows": [10],
        "min_periods": 4,
        "history_lag": 1,
        "robust": True,
    }
    cutoff = 52
    shocked = frame.copy()
    shocked.iloc[cutoff:] *= -500.0

    prefix = build_features(frame.iloc[:cutoff], config)
    extended = build_features(shocked, config).iloc[:cutoff]

    pd.testing.assert_frame_equal(prefix, extended, check_exact=False, rtol=1e-12, atol=1e-12)
    assert prefix["release"].iloc[:2].isna().all()
    pd.testing.assert_series_equal(
        prefix["release"].iloc[2:],
        frame["release"].iloc[:cutoff].shift(2).iloc[2:],
        check_names=False,
    )
    directly_lagged = apply_release_lags(frame[["release"]], {"release": 2})
    pd.testing.assert_series_equal(directly_lagged["release"], frame["release"].shift(2))


def test_factor_construction_is_prefix_invariant(market_frame: pd.DataFrame) -> None:
    config = {
        "normalization": "robust_zscore",
        "normalization_window": 40,
        "min_periods": 8,
        "history_lag": 1,
        "wheat_momentum_window": 10,
        "include_raw": True,
    }
    cutoff = 125
    shocked = market_frame.copy()
    shocked.iloc[cutoff:] = shocked.iloc[cutoff:] * np.linspace(
        25.0, 80.0, len(shocked) - cutoff
    )[:, None]

    prefix = build_mvp_factors(market_frame.iloc[:cutoff], config)
    extended = build_mvp_factors(shocked, config).iloc[:cutoff]

    pd.testing.assert_frame_equal(prefix, extended, check_exact=False, rtol=1e-12, atol=1e-12)
    assert set(MVP_FACTOR_COLUMNS).issubset(prefix.columns)
    assert {f"raw__{name}" for name in MVP_FACTOR_COLUMNS}.issubset(prefix.columns)
    assert prefix.attrs["point_in_time_safe"] is True


def test_regime_factor_contributions_exactly_reproduce_raw_and_displayed_scores(
    standardized_factors: pd.DataFrame,
) -> None:
    result = score_regimes(
        standardized_factors,
        {
            "intercepts": {name: (position - 3) * 0.07 for position, name in enumerate(REGIME_NAMES)},
            "raw_clip": 0.75,
            "min_coverage": 1.0,
        },
    )
    contributions = result.contributions
    raw_sum = (
        contributions.groupby(["date", "regime"], sort=False)["raw_contribution"]
        .sum()
        .unstack("regime")
        .reindex(index=result.raw_scores.index, columns=result.raw_scores.columns)
    )
    point_sum = (
        contributions.groupby(["date", "regime"], sort=False)["score_point_contribution"]
        .sum()
        .unstack("regime")
        .reindex(index=result.scores.index, columns=result.scores.columns)
    )

    pd.testing.assert_frame_equal(
        raw_sum,
        result.raw_scores,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        point_sum + 50.0,
        result.scores,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.coverage.eq(1.0).all(axis=None)
    assert result.scores.ge(0.0).all(axis=None)
    assert result.scores.le(100.0).all(axis=None)


def test_transition_matrices_and_every_forecast_vector_sum_to_one() -> None:
    index = pd.bdate_range("2023-01-02", periods=36, name="date")
    labels = pd.Series(
        (["calm"] * 5 + ["shock"] * 3 + ["calm"] * 4) * 3,
        index=index,
        name="regime",
    )
    config = {
        "states": ["calm", "shock"],
        "horizons": {"1d": 1, "1w": 5},
        "smoothing": 0.5,
    }

    result = estimate_transition_probabilities(labels, config=config)

    for horizon in config["horizons"]:
        matrix = result.transition_matrices[horizon]
        history = result.forecast_history[horizon]
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(history.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert matrix.ge(0.0).all(axis=None) and matrix.le(1.0).all(axis=None)
        assert history.ge(0.0).all(axis=None) and history.le(1.0).all(axis=None)
        assert result.transition_counts[horizon].ge(0).all(axis=None)
    assert set(result.latest_forecasts.columns) == {"horizon", "next_regime", "probability"}


def test_empirical_transition_forecasts_are_prefix_invariant() -> None:
    index = pd.bdate_range("2023-04-03", periods=30, name="date")
    labels = pd.Series(
        ["calm", "calm", "shock", "shock", "calm"] * 6,
        index=index,
    )
    config = {
        "states": ["calm", "shock"],
        "horizons": {"2d": 2},
        "smoothing": 1.0,
    }
    cutoff = 21

    prefix = estimate_transition_probabilities(labels.iloc[:cutoff], config=config)
    extended = estimate_transition_probabilities(labels, config=config)

    pd.testing.assert_frame_equal(
        prefix.forecast_history["2d"],
        extended.forecast_history["2d"].iloc[:cutoff],
    )
