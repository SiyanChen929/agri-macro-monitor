"""Research-only proportional-cost band around the frozen tanh signal.

The frictionless utility has derivative mu - sigma * atanh(weight), so its
optimum is the existing tanh(mu / sigma) target. A proportional one-way cost
creates a no-trade interval. The 5/21 horizon conversion is an explicit equal
daily accrual assumption, not an estimated signal decay or an alpha claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_aware_target(
    forecast: float,
    training_volatility: float,
    previous_target: float,
    *,
    one_way_cost_bps: float = 7.0,
    target_sessions: int = 21,
    decision_sessions: int = 5,
    maximum_asset_weight: float = 1.0 / 3.0,
) -> float:
    values = [forecast, training_volatility, previous_target,
              one_way_cost_bps, maximum_asset_weight]
    if not np.isfinite(values).all():
        raise ValueError("cost-aware target requires finite inputs")
    if training_volatility <= 0 or one_way_cost_bps < 0:
        raise ValueError("volatility must be positive and cost nonnegative")
    if target_sessions <= 0 or decision_sessions <= 0:
        raise ValueError("horizons must be positive")
    if not 0 < maximum_asset_weight < 1:
        raise ValueError("asset cap must be strictly between zero and one")
    if abs(previous_target) > maximum_asset_weight + 1e-12:
        raise ValueError("previous target exceeds asset cap")
    cost = one_way_cost_bps / 10_000.0 * target_sessions / decision_sessions
    lower = np.clip(np.tanh((forecast - cost) / training_volatility),
                    -maximum_asset_weight, maximum_asset_weight)
    upper = np.clip(np.tanh((forecast + cost) / training_volatility),
                    -maximum_asset_weight, maximum_asset_weight)
    return float(np.clip(previous_target, lower, upper))


def apply_cost_aware_targets(
    signals: pd.DataFrame,
    *,
    one_way_cost_bps: float,
    maximum_asset_weight: float,
    target_sessions: int = 21,
    decision_sessions: int = 5,
) -> pd.DataFrame:
    """Use only each dated forecast and the preceding executable target.

    The caller passes the common evaluation window, so every asset begins in
    cash. Costs used to create the policy stay fixed across cost stress runs.
    The existing runner retains next-session execution and forced liquidation.
    """
    required = {"date", "asset", "prediction", "training_target_volatility"}
    if not required.issubset(signals):
        raise ValueError(f"missing signal fields: {sorted(required - set(signals))}")
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any() or out.duplicated(["date", "asset"]).any():
        raise ValueError("dated asset signals must be unique and complete")
    out = out.sort_values(["asset", "date"], kind="stable")
    for _, group in out.groupby("asset", sort=False):
        previous = 0.0
        for index, row in group.iterrows():
            previous = cost_aware_target(
                float(row["prediction"]), float(row["training_target_volatility"]),
                previous, one_way_cost_bps=one_way_cost_bps,
                target_sessions=target_sessions, decision_sessions=decision_sessions,
                maximum_asset_weight=maximum_asset_weight,
            )
            out.loc[index, "target_weight"] = previous
    return out.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
