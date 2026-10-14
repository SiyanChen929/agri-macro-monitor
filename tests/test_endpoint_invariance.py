"""Regression tests for endpoint-independent walk-forward forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.regimes import (
    MVP_FACTOR_COLUMNS,
    estimate_statistical_regimes,
)
from agri_macro_quant.transitions import estimate_transition_probabilities


def _factor_history(periods: int = 128) -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=periods, name="date")
    t = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            name: (
                np.sin(t / (4.0 + position))
                + 0.35 * np.cos(t / (7.0 + position))
                + 0.002 * (position + 1) * t
            )
            for position, name in enumerate(MVP_FACTOR_COLUMNS)
        },
        index=index,
    )


def test_statistical_regime_probabilities_are_strictly_prefix_invariant() -> None:
    factors = _factor_history()
    states = ["calm", "shock", "recovery"]
    labels = pd.Series(
        [states[(position // 7 + position // 19) % len(states)] for position in range(len(factors))],
        index=factors.index,
        name="regime",
    )
    cutoff = 75
    extended_factors = factors.copy()
    extended_factors.iloc[cutoff:] = (
        extended_factors.iloc[cutoff:].iloc[::-1].to_numpy() * -10_000.0
    )
    extended_labels = labels.copy()
    extended_labels.iloc[cutoff:] = np.resize(
        np.array(["recovery", "shock", "calm"], dtype=object),
        len(labels) - cutoff,
    )
    config = {
        "classes": states,
        "min_train_size": 30,
        "min_class_observations": 5,
        "refit_every": 17,
        "transparent_weight": 0.0,
        "max_iter": 60,
        "random_seed": 314,
    }

    prefix = estimate_statistical_regimes(
        factors.iloc[:cutoff], labels.iloc[:cutoff], config
    )
    extended = estimate_statistical_regimes(
        extended_factors, extended_labels, config
    )

    pd.testing.assert_frame_equal(
        prefix.probabilities,
        extended.probabilities.iloc[:cutoff],
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    # Fits occur at positions 30, 47, and 64; position 74 is only the
    # endpoint of this particular slice and must not trigger another fit.
    assert prefix.metadata["last_fit_position"] == 64
    assert prefix.metadata["last_fit_date"] == factors.index[64]
    last_scheduled = estimate_statistical_regimes(
        factors.iloc[:65], labels.iloc[:65], config
    )
    pd.testing.assert_frame_equal(prefix.coefficients, last_scheduled.coefficients)
    pd.testing.assert_series_equal(prefix.intercepts, last_scheduled.intercepts)


def test_conditional_transition_forecasts_are_strictly_prefix_invariant() -> None:
    factors = _factor_history(140)
    states = ["calm", "shock", "recovery"]
    labels = pd.Series(
        [states[(position // 5 + position // 17) % len(states)] for position in range(len(factors))],
        index=factors.index,
        name="regime",
    )
    transition_features = factors[["weather_stress", "policy_stress", "corn_urea"]]
    cutoff = 72
    extended_features = transition_features.copy()
    extended_features.iloc[cutoff:] = (
        50_000.0 + np.arange(len(extended_features) - cutoff)[:, None]
    )
    extended_labels = labels.copy()
    extended_labels.iloc[cutoff:] = np.resize(
        np.array(["shock", "recovery", "calm"], dtype=object),
        len(labels) - cutoff,
    )
    config = {
        "states": states,
        "horizons": {"3d": 3},
        "min_train_size": 24,
        "min_class_observations": 4,
        "refit_every": 13,
        "empirical_weight": 0.2,
        "max_iter": 60,
        "random_seed": 2718,
    }

    prefix = estimate_transition_probabilities(
        labels.iloc[:cutoff], transition_features.iloc[:cutoff], config
    )
    extended = estimate_transition_probabilities(
        extended_labels, extended_features, config
    )

    pd.testing.assert_frame_equal(
        prefix.forecast_history["3d"],
        extended.forecast_history["3d"].iloc[:cutoff],
        check_exact=False,
        rtol=1e-14,
        atol=1e-14,
    )
    # With a three-day target the first fit is position 26, followed by 39,
    # 52, and 65.  Position 71 is an arbitrary slice endpoint, not a fit date.
    model_metadata = prefix.metadata["models"]["3d"]
    assert model_metadata["last_fit_position"] == 65
    assert model_metadata["last_fit_date"] == labels.index[65]
    last_scheduled = estimate_transition_probabilities(
        labels.iloc[:66], transition_features.iloc[:66], config
    )
    pd.testing.assert_frame_equal(
        prefix.model_coefficients["3d"],
        last_scheduled.model_coefficients["3d"],
    )
