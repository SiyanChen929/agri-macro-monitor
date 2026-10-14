"""Portfolio risk decomposition, exposure controls, and drawdown stops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DRAWDOWN_SCHEDULE: tuple[tuple[float, float], ...] = (
    (0.05, 0.80),
    (0.10, 0.50),
    (0.15, 0.20),
)


def repair_covariance(
    covariance: pd.DataFrame,
    assets: pd.Index | Sequence[str] | None = None,
    *,
    fallback_variance: float = 0.25**2,
    eigenvalue_floor: float = 1e-8,
) -> pd.DataFrame:
    """Return a finite, symmetric positive-definite covariance estimate.

    The repair operates on the implied correlation matrix, clips its
    eigenvalues, and rescales it back to covariance units.  This preserves all
    finite positive input variances, unlike clipping covariance eigenvalues
    directly.  A small relative eigenvalue floor prevents an invalid supplied
    matrix from turning a non-zero portfolio into silently reported zero risk.

    Repair diagnostics are attached to ``result.attrs["repair_diagnostics"]``.
    They distinguish ordinary PSD input from data cleaning, symmetrization, and
    spectral repair.
    """

    if not isinstance(covariance, pd.DataFrame):
        covariance = pd.DataFrame(covariance)
    labels = pd.Index(covariance.index if assets is None else assets).astype(str)
    raw = covariance.copy()
    raw.index = raw.index.astype(str)
    raw.columns = raw.columns.astype(str)
    raw = raw.apply(pd.to_numeric, errors="coerce").reindex(index=labels, columns=labels)

    values = raw.to_numpy(dtype=float, copy=True)
    invalid_mask = ~np.isfinite(values)
    invalid_count = int(invalid_mask.sum())
    values[invalid_mask] = np.nan
    asymmetry = values - values.T
    finite_asymmetry = np.abs(asymmetry[np.isfinite(asymmetry)])
    max_asymmetry = float(finite_asymmetry.max()) if finite_asymmetry.size else 0.0

    # An observed value on either side is preferable to replacing both with
    # zero when a triangular covariance matrix is supplied.
    transpose = values.T
    paired = np.where(
        np.isfinite(values) & np.isfinite(transpose),
        (values + transpose) / 2.0,
        np.where(np.isfinite(values), values, np.where(np.isfinite(transpose), transpose, 0.0)),
    )
    variance_fallback = max(float(fallback_variance), np.finfo(float).eps)
    diagonal = np.diag(paired).copy()
    bad_diagonal = ~np.isfinite(diagonal) | (diagonal <= 0.0)
    diagonal[bad_diagonal] = variance_fallback
    np.fill_diagonal(paired, diagonal)

    scale = np.sqrt(diagonal)
    correlation = paired / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    min_before = float(eigenvalues.min()) if len(eigenvalues) else 0.0
    max_before = float(eigenvalues.max()) if len(eigenvalues) else 0.0

    # Interpret the configured floor relative to the covariance scale.  A
    # strictly positive default eliminates null-risk artifacts caused by an
    # indefinite input while remaining immaterial for ordinary estimates.
    relative_floor = max(0.0, float(eigenvalue_floor))
    floor = max(np.finfo(float).eps, relative_floor * max(1.0, max_before))
    clipped = np.maximum(eigenvalues, floor)
    clipped_count = int(np.count_nonzero(eigenvalues < floor))
    if clipped_count:
        correlation = (eigenvectors * clipped) @ eigenvectors.T
        repaired_diagonal = np.sqrt(np.maximum(np.diag(correlation), np.finfo(float).eps))
        correlation = correlation / np.outer(repaired_diagonal, repaired_diagonal)
        correlation = (correlation + correlation.T) / 2.0
        np.fill_diagonal(correlation, 1.0)

    repaired_values = correlation * np.outer(scale, scale)
    repaired_values = (repaired_values + repaired_values.T) / 2.0
    np.fill_diagonal(repaired_values, diagonal)
    min_after = float(np.linalg.eigvalsh(repaired_values).min()) if len(labels) else 0.0
    diagnostics: dict[str, Any] = {
        "method": "correlation_eigenvalue_clipping",
        "repaired": bool(
            invalid_count
            or bad_diagonal.any()
            or max_asymmetry > 1e-12
            or clipped_count
        ),
        "invalid_values_replaced": invalid_count,
        "diagonal_values_replaced": int(bad_diagonal.sum()),
        "max_asymmetry_before": max_asymmetry,
        "min_correlation_eigenvalue_before": min_before,
        "max_correlation_eigenvalue_before": max_before,
        "eigenvalues_clipped": clipped_count,
        "relative_eigenvalue_floor": relative_floor,
        "min_covariance_eigenvalue_after": min_after,
        "diagonal_preserved": bool(
            np.allclose(np.diag(repaired_values), diagonal, rtol=0.0, atol=1e-12)
        ),
    }
    result = pd.DataFrame(repaired_values, index=labels, columns=labels)
    result.attrs["repair_diagnostics"] = diagnostics
    return result


def wealth_index(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compound simple returns while treating missing observations as flat."""

    series = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    result = float(initial_value) * (1.0 + series).cumprod()
    result.name = "equity"
    # Keep the pre-return NAV available to downstream drawdown calculations
    # without inserting a synthetic row that would change the caller's index.
    result.attrs["initial_value"] = float(initial_value)
    return result


def drawdown_series(equity_or_returns: pd.Series, input_type: str = "auto") -> pd.Series:
    """Return drawdown from the running peak, aligned to the supplied index.

    The pre-first-observation NAV is part of the peak history. Return inputs
    use an initial NAV of 1.0. Equity inputs may carry their actual starting
    capital in ``Series.attrs["initial_value"]``; otherwise they are treated as
    a wealth index whose pre-observation NAV was 1.0. This captures a loss on
    the first row without exposing a synthetic baseline row in the result.
    """

    if not isinstance(equity_or_returns, pd.Series):
        equity_or_returns = pd.Series(equity_or_returns, dtype=float)
    source_attrs = dict(equity_or_returns.attrs)
    series = pd.to_numeric(equity_or_returns, errors="coerce")
    kind = str(input_type).lower()
    if kind == "auto":
        finite = series.dropna()
        kind = "equity" if (not finite.empty and (finite > 0).all() and finite.median() > 0.25) else "returns"
    if kind in {"return", "returns", "pnl"}:
        equity = wealth_index(series)
        initial_value = 1.0
    elif kind in {"equity", "wealth", "nav"}:
        equity = series.ffill()
        initial_value = float(
            source_attrs.get("initial_value", source_attrs.get("initial_nav", 1.0))
        )
    else:
        raise ValueError("input_type must be 'auto', 'returns', or 'equity'")
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial NAV must be finite and positive")
    # Clipping the running peak to the pre-return NAV is algebraically
    # equivalent to prepending that NAV, but preserves exact index alignment.
    peak = equity.cummax().clip(lower=initial_value).replace(0.0, np.nan)
    result = (equity / peak - 1.0).fillna(0.0)
    result.name = "drawdown"
    return result


def _parse_drawdown_schedule(config: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    cfg = dict(config or {})
    if isinstance(cfg.get("portfolio"), Mapping):
        cfg = {**cfg, **dict(cfg["portfolio"])}
    raw = cfg.get(
        "drawdown_schedule",
        cfg.get("drawdown_controls", cfg.get("thresholds", DEFAULT_DRAWDOWN_SCHEDULE)),
    )
    schedule: list[tuple[float, float]] = []
    if isinstance(raw, Mapping):
        iterator = raw.items()
    else:
        iterator = raw
    for item in iterator:
        if isinstance(item, Mapping):
            threshold = item.get("drawdown", item.get("threshold"))
            multiplier = item.get("risk_multiplier", item.get("multiplier"))
        else:
            threshold, multiplier = item
        threshold_value = abs(float(threshold))
        multiplier_value = float(np.clip(float(multiplier), 0.0, 1.0))
        schedule.append((threshold_value, multiplier_value))
    return sorted(schedule, key=lambda pair: pair[0])


def drawdown_risk_multiplier(drawdown: float, config: Mapping[str, Any] | None = None) -> float:
    """Map a current drawdown to its configured portfolio-risk multiplier."""

    loss = abs(min(0.0, float(drawdown)))
    multiplier = 1.0
    for threshold, scheduled_multiplier in _parse_drawdown_schedule(config):
        if loss >= threshold:
            multiplier = min(multiplier, scheduled_multiplier)
    return float(multiplier)


def apply_drawdown_control(
    weights: pd.Series | pd.DataFrame,
    equity_or_returns: pd.Series,
    config: Mapping[str, Any] | None = None,
    return_details: bool = False,
) -> pd.Series | pd.DataFrame | tuple[pd.Series | pd.DataFrame, dict[str, float]]:
    """Scale current weights according to the latest realized drawdown.

    The control uses only the supplied equity history through its last point;
    a backtest should therefore call it before calculating the next return.
    """

    cfg = dict(config or {})
    drawdowns = drawdown_series(equity_or_returns, str(cfg.get("input_type", "auto")))
    latest = float(drawdowns.dropna().iloc[-1]) if not drawdowns.dropna().empty else 0.0
    multiplier = drawdown_risk_multiplier(latest, cfg)
    controlled = weights.astype(float) * multiplier
    details = {"drawdown": latest, "risk_multiplier": multiplier}
    if return_details:
        return controlled, details
    return controlled


def apply_signal_regime_stops(
    weights: pd.Series,
    signals: pd.Series | Mapping[str, float] | None = None,
    current_regimes: str | Sequence[str] | None = None,
    expected_regimes: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Apply transparent signal deterioration and regime-transition stops."""

    cfg = dict(config or {})
    result = weights.astype(float).copy()
    minimum = max(0.0, float(cfg.get("signal_stop_threshold", 0.10)))
    deterioration_multiplier = float(np.clip(float(cfg.get("signal_stop_multiplier", 0.0)), 0.0, 1.0))
    if signals is not None:
        signal_series = pd.Series(signals, dtype=float).reindex(result.index)
        observed = signal_series.notna()
        deteriorated = observed & (signal_series.abs() < minimum)
        # A sign reversal is stronger evidence than simple weakening.
        reversed_direction = observed & (np.sign(signal_series) != np.sign(result))
        result.loc[deteriorated | reversed_direction] *= deterioration_multiplier

    if expected_regimes:
        active = {str(current_regimes)} if isinstance(current_regimes, str) else set(map(str, current_regimes or []))
        failure_multiplier = float(np.clip(float(cfg.get("regime_stop_multiplier", 0.0)), 0.0, 1.0))
        for asset, expected in expected_regimes.items():
            if asset in result.index and str(expected) not in active:
                result.loc[asset] *= failure_multiplier
    return result


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Return marginal and component contributions to annualized volatility."""

    weight = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    cov = repair_covariance(covariance, weight.index)
    vector = weight.to_numpy()
    marginal_variance = cov.to_numpy() @ vector
    variance = float(vector @ marginal_variance)
    volatility = np.sqrt(max(variance, 0.0))
    if volatility > 0:
        marginal_vol = marginal_variance / volatility
        component = vector * marginal_vol
        percent = component / volatility
    else:
        marginal_vol = np.zeros_like(vector)
        component = np.zeros_like(vector)
        percent = np.zeros_like(vector)
    result = pd.DataFrame(
        {
            "weight": vector,
            "marginal_volatility": marginal_vol,
            "risk_contribution": component,
            "risk_contribution_pct": percent,
        },
        index=weight.index,
    )
    result.attrs["covariance_repair"] = dict(cov.attrs.get("repair_diagnostics", {}))
    result.attrs["expected_volatility"] = float(volatility)
    return result


def factor_overlap(
    weights: pd.Series,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Show signed and gross exposure to each economic risk factor."""

    if isinstance(factor_exposures, Mapping):
        exposure = pd.DataFrame.from_dict(factor_exposures, orient="index")
    else:
        exposure = factor_exposures.copy()
    exposure = exposure.apply(pd.to_numeric, errors="coerce").reindex(weights.index).fillna(0.0)
    signed = exposure.mul(weights, axis=0).sum(axis=0)
    gross = exposure.abs().mul(weights.abs(), axis=0).sum(axis=0)
    concentration = signed.abs().div(gross.replace(0.0, np.nan)).fillna(0.0)
    result = pd.DataFrame(
        {
            "signed_exposure": signed,
            "gross_overlap": gross,
            "directional_concentration": concentration,
        }
    )
    result.index.name = "factor"
    return result.sort_values("gross_overlap", ascending=False)


def _estimate_covariance(
    assets: pd.Index,
    returns: pd.DataFrame | None,
    covariance: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    annualization = float(config.get("annualization", 252))
    if covariance is not None:
        result = covariance.copy().apply(pd.to_numeric, errors="coerce")
        result = result.reindex(index=assets, columns=assets)
        if not bool(config.get("covariance_is_annualized", True)):
            result *= annualization
    elif returns is not None and not returns.empty:
        lookback = max(2, int(config.get("lookback", 60)))
        result = returns.reindex(columns=assets).tail(lookback).cov() * annualization
    else:
        fallback = float(config.get("fallback_volatility", 0.25))
        result = pd.DataFrame(np.eye(len(assets)) * fallback**2, index=assets, columns=assets)
    fallback_variance = float(config.get("fallback_volatility", 0.25)) ** 2
    return repair_covariance(
        result,
        assets,
        fallback_variance=fallback_variance,
        eigenvalue_floor=float(config.get("covariance_eigenvalue_floor", 1e-8)),
    )


def portfolio_risk_report(
    weights: pd.Series,
    returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate exposure, covariance risk, historical VaR/CVaR, and overlap."""

    cfg = dict(config or {})
    if isinstance(cfg.get("risk"), Mapping):
        cfg = {**cfg, **dict(cfg["risk"])}
    if isinstance(cfg.get("portfolio"), Mapping):
        cfg = {**cfg, **dict(cfg["portfolio"])}
    weight = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    weight.index = weight.index.astype(str)
    cov = _estimate_covariance(weight.index, returns, covariance, cfg)
    contributions = risk_contributions(weight, cov)
    expected_vol = float(contributions["risk_contribution"].sum())
    # Negative component contributions are valid hedges; covariance formula is
    # the authoritative portfolio volatility.
    expected_vol = float(np.sqrt(max(0.0, weight.to_numpy() @ cov.to_numpy() @ weight.to_numpy())))

    portfolio_returns = pd.Series(dtype=float)
    priced_weight_coverage = pd.Series(dtype=float)
    excluded_unpriced_days = 0
    if returns is not None and not returns.empty:
        aligned = returns.reindex(columns=weight.index).apply(
            pd.to_numeric, errors="coerce"
        )
        gross_weight = float(weight.abs().sum())
        if gross_weight > 0.0:
            priced_weight_coverage = aligned.notna().mul(
                weight.abs(), axis=1
            ).sum(axis=1) / gross_weight
        else:
            priced_weight_coverage = pd.Series(1.0, index=aligned.index)
        minimum_coverage = float(
            np.clip(cfg.get("min_priced_weight_coverage", 1.0), 0.0, 1.0)
        )
        eligible = priced_weight_coverage.ge(minimum_coverage - 1e-12)
        excluded_unpriced_days = int((~eligible).sum())
        portfolio_returns = aligned.mul(weight, axis=1).sum(
            axis=1, min_count=1
        ).where(eligible).dropna()
    confidence = float(cfg.get("var_confidence", 0.95))
    confidence = float(np.clip(confidence, 0.50, 0.999))
    if not portfolio_returns.dropna().empty:
        quantile = float(portfolio_returns.quantile(1.0 - confidence))
        tail = portfolio_returns.loc[portfolio_returns <= quantile]
        historical_var = max(0.0, -quantile)
        historical_cvar = max(0.0, -float(tail.mean())) if not tail.empty else historical_var
    else:
        historical_var = np.nan
        historical_cvar = np.nan

    overlap = factor_overlap(weight, factor_exposures) if factor_exposures is not None else pd.DataFrame()
    largest_risk = (
        str(contributions["risk_contribution"].abs().idxmax())
        if not contributions.empty and contributions["risk_contribution"].abs().max() > 0
        else None
    )
    report: dict[str, Any] = {
        "gross_exposure": float(weight.abs().sum()),
        "net_exposure": float(weight.sum()),
        "long_exposure": float(weight.clip(lower=0.0).sum()),
        "short_exposure": float(-weight.clip(upper=0.0).sum()),
        "expected_volatility": expected_vol,
        "historical_var": historical_var,
        "historical_cvar": historical_cvar,
        "var_confidence": confidence,
        "historical_priced_weight_coverage_mean": (
            float(priced_weight_coverage.mean())
            if not priced_weight_coverage.empty
            else np.nan
        ),
        "historical_priced_weight_coverage_min": (
            float(priced_weight_coverage.min())
            if not priced_weight_coverage.empty
            else np.nan
        ),
        "historical_unpriced_days_excluded": excluded_unpriced_days,
        "historical_risk_observations": int(len(portfolio_returns)),
        "largest_risk_contributor": largest_risk,
        "risk_contributions": contributions,
        "factor_overlap": overlap,
        "covariance_repair": dict(cov.attrs.get("repair_diagnostics", {})),
    }
    if not portfolio_returns.empty:
        report["static_weight_replay_drawdown"] = float(
            drawdown_series(portfolio_returns, "returns").iloc[-1]
        )
    else:
        report["static_weight_replay_drawdown"] = np.nan
    # A static replay of today's weights is not the live strategy drawdown.
    # Callers may supply the actual strategy value through configuration.
    report["current_drawdown"] = float(cfg.get("current_drawdown", np.nan))
    return report


# Common aliases.
compute_drawdown = drawdown_series
risk_report = portfolio_risk_report


__all__ = [
    "DEFAULT_DRAWDOWN_SCHEDULE",
    "apply_drawdown_control",
    "apply_signal_regime_stops",
    "compute_drawdown",
    "drawdown_risk_multiplier",
    "drawdown_series",
    "factor_overlap",
    "portfolio_risk_report",
    "repair_covariance",
    "risk_contributions",
    "risk_report",
    "wealth_index",
]
