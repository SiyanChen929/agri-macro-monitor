"""Live model-health, drift, stability, and performance monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def model_health_report(
    factors: pd.DataFrame,
    regime_scores: pd.DataFrame,
    *,
    strategy_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    data_quality_score: float | None = None,
    recent_window: int = 63,
) -> dict[str, Any]:
    """Summarize factor drift and simulated backtest-tail behavior causally.

    ``strategy_returns`` are treated as research backtest observations.  The
    recent tail is compared with the disjoint prior sample, never described as
    realized live performance and never benchmarked against a history that
    contains the same recent observations.
    """

    factor_frame = factors.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    scores = regime_scores.apply(pd.to_numeric, errors="coerce")
    recent_window = max(10, int(recent_window))
    historical = factor_frame.iloc[:-recent_window] if len(factor_frame) > recent_window else factor_frame.iloc[:0]
    recent = factor_frame.tail(recent_window)
    drift_rows: list[dict[str, Any]] = []
    if not historical.empty:
        for column in factor_frame.columns:
            scale = float(historical[column].std(ddof=0))
            shift = float(recent[column].mean() - historical[column].mean())
            standardized = shift / scale if np.isfinite(scale) and scale > 1e-12 else np.nan
            drift_rows.append(
                {
                    "factor": str(column),
                    "historical_mean": float(historical[column].mean()),
                    "recent_mean": float(recent[column].mean()),
                    "standardized_mean_shift": standardized,
                }
            )
    drift = pd.DataFrame(drift_rows)
    if not drift.empty:
        drift = drift.sort_values("standardized_mean_shift", key=lambda s: s.abs(), ascending=False)

    dominant = scores.fillna(-np.inf).idxmax(axis=1) if not scores.empty else pd.Series(dtype=str)
    recent_labels = dominant.tail(recent_window)
    switches = int((recent_labels != recent_labels.shift(1)).sum() - (1 if len(recent_labels) else 0))
    stability = 1.0 - switches / max(1, len(recent_labels) - 1)
    warnings: list[str] = []
    if not drift.empty and drift["standardized_mean_shift"].abs().max() > 2.5:
        factor = str(drift.iloc[0]["factor"])
        warnings.append(f"Feature distribution shift exceeds 2.5σ for {factor}.")
    if stability < 0.70:
        warnings.append("Dominant-regime classification is unstable over the recent window.")

    performance: dict[str, float] = {}
    if strategy_returns is not None:
        returns = pd.to_numeric(strategy_returns, errors="coerce").dropna()
        if len(returns) >= 20:
            recent_returns = returns.tail(recent_window)
            prior_returns = returns.iloc[: max(0, len(returns) - len(recent_returns))]

            def sharpe(series: pd.Series) -> float:
                vol = float(series.std(ddof=1))
                return float(np.sqrt(252.0) * series.mean() / vol) if vol > 0 else 0.0

            performance = {
                "scope": "simulated_backtest_tail_vs_disjoint_prior_sample",
                "recent_backtest_sharpe": sharpe(recent_returns),
                "recent_backtest_return": float((1.0 + recent_returns).prod() - 1.0),
                "recent_observations": float(len(recent_returns)),
            }
            if len(prior_returns) >= 20:
                performance["prior_backtest_sharpe"] = sharpe(prior_returns)
                performance["prior_observations"] = float(len(prior_returns))
                if (
                    performance["recent_backtest_sharpe"]
                    < performance["prior_backtest_sharpe"] - 2.0
                ):
                    warnings.append(
                        "The recent simulated backtest tail is materially below the disjoint prior sample."
                    )
    latest_turnover = float(pd.to_numeric(turnover, errors="coerce").tail(21).mean()) if turnover is not None and len(turnover) else np.nan
    return {
        "status": "warning" if warnings else "healthy",
        "data_quality_score": data_quality_score,
        "regime_stability": float(np.clip(stability, 0.0, 1.0)),
        "recent_regime_switches": switches,
        "recent_average_turnover": latest_turnover,
        "performance": performance,
        "factor_drift": drift,
        "warnings": warnings,
    }


__all__ = ["model_health_report"]
