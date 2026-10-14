import numpy as np
import pandas as pd
import pytest

from agri_macro_quant.research.cost_aware_execution import (
    apply_cost_aware_targets, cost_aware_target,
)


def test_zero_cost_recovers_original_signal_and_cost_band_solves_utility():
    grid = np.linspace(-1 / 3, 1 / 3, 20001)
    penalty = grid * np.arctanh(grid) + 0.5 * np.log(1 - grid**2)
    for mu in [-0.04, -0.003, 0, 0.002, 0.04]:
        for old in [-0.3, 0, 0.25]:
            target = cost_aware_target(mu, 0.07, old, one_way_cost_bps=0)
            assert target == pytest.approx(np.clip(np.tanh(mu / 0.07), -1/3, 1/3))
            candidate = cost_aware_target(mu, 0.07, old)
            utility = (5 / 21) * (mu * grid - 0.07 * penalty) - 0.0007 * np.abs(grid - old)
            assert abs(candidate - grid[np.argmax(utility)]) < 4e-5


def test_future_forecasts_cannot_change_past_targets():
    signals = pd.DataFrame({"date": pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
                            "asset": "ZC", "prediction": [0.001, 0.02, -0.01, 0.03],
                            "training_target_volatility": 0.08})
    kwargs = {"one_way_cost_bps": 7, "maximum_asset_weight": 1/3}
    full = apply_cost_aware_targets(signals, **kwargs)
    prefix = apply_cost_aware_targets(signals.iloc[:2], **kwargs)
    np.testing.assert_allclose(full.target_weight.iloc[:2], prefix.target_weight)
    assert full.target_weight.iloc[0] == 0
    assert abs(full.target_weight).max() <= 1/3
    with pytest.raises(ValueError, match="unique"):
        apply_cost_aware_targets(pd.concat([signals, signals.iloc[:1]]), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        cost_aware_target(np.nan, 0.1, 0)
