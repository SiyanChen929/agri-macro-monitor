"""Conservative signal-to-position portfolio construction.

No external optimizer is required.  The implementation intentionally favors
transparent sequential constraints over a fragile black-box optimum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .risk import repair_covariance


DEFAULT_PORTFOLIO_CONFIG: dict[str, Any] = {
    "annualization": 252,
    "vol_lookback": 60,
    "min_volatility": 0.05,
    "fallback_volatility": 0.25,
    "target_volatility": 0.08,
    "max_position_weight": 0.15,
    "max_gross_exposure": 1.00,
    "max_net_exposure": 0.35,
    "max_asset_class_exposure": 0.40,
    "max_commodity_exposure": 0.35,
    "commodity_asset_classes": ("grain", "energy"),
    "max_fertilizer_equity_exposure": 0.30,
    "max_factor_exposure": 0.35,
    "confidence_floor": 0.0,
    "min_signal": 0.05,
    "min_return_observations": 20,
    "require_latest_return": True,
}


DEFAULT_ASSET_CLASSES: dict[str, str] = {
    "ZW": "grain",
    "KE": "grain",
    "ZC": "grain",
    "ZS": "grain",
    "ZM": "grain",
    "ZL": "grain",
    "NG": "energy",
    "CL": "energy",
    "BRENT": "energy",
    "CF": "fertilizer_equity",
    "NTR": "fertilizer_equity",
    "MOS": "fertilizer_equity",
    "UAN": "fertilizer_equity",
    "IPI": "fertilizer_equity",
    "CTVA": "ag_equity",
    "DXY": "fx",
    "US2Y": "rates",
    "US10Y": "rates",
}


DEFAULT_ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "ZW": ("ZW", "wheat", "ZW=F"),
    "ZC": ("ZC", "corn", "ZC=F"),
    "ZS": ("ZS", "soybeans", "soybean", "ZS=F"),
    "ZM": ("ZM", "soybean_meal", "ZM=F"),
    "ZL": ("ZL", "soybean_oil", "ZL=F"),
    "NG": ("NG", "natural_gas", "NG=F"),
    "CL": ("CL", "oil", "wti", "CL=F"),
    "CF": ("CF", "cf"),
    "NTR": ("NTR", "ntr"),
    "MOS": ("MOS", "mos"),
    "UAN": ("UAN", "uan"),
    "IPI": ("IPI", "ipi"),
    "CTVA": ("CTVA", "ctva"),
    "DXY": ("DXY", "dxy", "DX-Y.NYB"),
    "US2Y": ("US2Y", "us2y", "us_2y"),
    "US10Y": ("US10Y", "us10y", "us_10y"),
}


def _portfolio_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    outer = dict(config or {})
    nested = dict(outer.get("portfolio", {})) if isinstance(outer.get("portfolio"), Mapping) else {}
    supplied = {**outer, **nested}
    cfg = {**DEFAULT_PORTFOLIO_CONFIG, **supplied}
    aliases = {
        "annualization_days": "annualization",
        "max_gross_leverage": "max_gross_exposure",
        "max_asset_class_weight": "max_asset_class_exposure",
        "max_commodity_weight": "max_commodity_exposure",
        "max_fertilizer_equity_weight": "max_fertilizer_equity_exposure",
    }
    for source, target in aliases.items():
        if source in supplied and target not in supplied:
            cfg[target] = supplied[source]
    return cfg


def _latest_signal_vector(signals: pd.DataFrame | pd.Series) -> tuple[pd.Series, pd.Series]:
    """Extract one signal and confidence observation per instrument."""

    if isinstance(signals, pd.Series):
        signal = pd.to_numeric(signals, errors="coerce").fillna(0.0)
        confidence = pd.Series(1.0, index=signal.index)
        return signal.clip(-1.0, 1.0), confidence
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame or Series")

    frame = signals.copy()
    if {"asset", "signal"}.issubset(frame.columns):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            latest_date = frame["date"].max()
            frame = frame.loc[frame["date"] == latest_date]
        raw_signal = pd.to_numeric(frame["signal"], errors="coerce").fillna(0.0)
        raw_confidence = (
            pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
            if "confidence" in frame.columns
            else pd.Series(1.0, index=frame.index)
        )
        # Expand relative-value records into tradable legs.  Three statistical
        # methods for one pair remain an ensemble through the final group mean.
        if {"long_leg", "short_leg"}.issubset(frame.columns):
            is_pair = frame["long_leg"].notna() & frame["short_leg"].notna()
            absolute = frame.loc[~is_pair, ["asset"]].assign(
                _asset=frame.loc[~is_pair, "asset"].astype(str),
                _signal=raw_signal.loc[~is_pair],
                _confidence=raw_confidence.loc[~is_pair],
            )
            beta = (
                pd.to_numeric(frame.loc[is_pair, "hedge_ratio"], errors="coerce").fillna(1.0)
                if "hedge_ratio" in frame.columns
                else pd.Series(1.0, index=frame.index[is_pair])
            )
            long_rows = pd.DataFrame(
                {
                    "_asset": frame.loc[is_pair, "long_leg"].astype(str),
                    "_signal": raw_signal.loc[is_pair],
                    "_confidence": raw_confidence.loc[is_pair],
                }
            )
            short_rows = pd.DataFrame(
                {
                    "_asset": frame.loc[is_pair, "short_leg"].astype(str),
                    # The beta-neutral residual is r_long - beta*r_short.
                    # Preserve beta's sign: a negative beta correctly makes
                    # both executable legs point in the same direction.
                    "_signal": -raw_signal.loc[is_pair] * beta,
                    "_confidence": raw_confidence.loc[is_pair],
                }
            )
            expanded = pd.concat(
                [absolute[["_asset", "_signal", "_confidence"]], long_rows, short_rows],
                ignore_index=True,
            )
        else:
            expanded = pd.DataFrame(
                {
                    "_asset": frame["asset"].astype(str),
                    "_signal": raw_signal,
                    "_confidence": raw_confidence,
                }
            )
        normalized_confidence = pd.to_numeric(
            expanded["_confidence"], errors="coerce"
        ).fillna(0.0)
        finite = normalized_confidence[np.isfinite(normalized_confidence)]
        if not finite.empty and finite.abs().quantile(0.95) > 1.5:
            normalized_confidence = normalized_confidence / 100.0
        expanded["_confidence"] = normalized_confidence.clip(0.0, 1.0)
        expanded["_effective_evidence"] = (
            pd.to_numeric(expanded["_signal"], errors="coerce").fillna(0.0)
            * expanded["_confidence"]
        )
        grouped = expanded.groupby("_asset", sort=True)
        confidence = grouped["_confidence"].mean()
        effective_evidence = grouped["_effective_evidence"].mean()
        # Keep the public signal/confidence decomposition while guaranteeing
        # that downstream ``signal * confidence`` equals the ensemble's mean
        # confidence-weighted signed evidence. A low-confidence opposing view
        # therefore cannot cancel a high-confidence view at equal strength.
        signal = effective_evidence.div(confidence.replace(0.0, np.nan)).fillna(0.0)
        return signal.clip(-1.0, 1.0), confidence.clip(0.0, 1.0)

    # A single wide row is accepted for ergonomic notebook use.
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    signal = numeric.iloc[-1].fillna(0.0).clip(-1.0, 1.0)
    return signal, pd.Series(1.0, index=signal.index)


def aggregate_executable_signals(
    signals: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """Return the current underlying-asset signal and confidence vectors.

    Relative-value rows are expanded using their signed hedge ratios, exactly
    as they are for portfolio construction.  This is useful for transparent
    live signal-stop checks and diagnostics.
    """

    signal, confidence = _latest_signal_vector(signals)
    return pd.DataFrame(
        {
            "signal": signal,
            "confidence": confidence.reindex(signal.index).fillna(0.0),
        }
    ).sort_index()


def _covariance_and_volatility(
    assets: pd.Index,
    returns: pd.DataFrame | None,
    covariance: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    annualization = float(config["annualization"])
    fallback = max(1e-6, float(config["fallback_volatility"]))
    min_vol = max(1e-6, float(config["min_volatility"]))

    if covariance is not None:
        cov = covariance.copy().apply(pd.to_numeric, errors="coerce")
        cov = cov.reindex(index=assets, columns=assets).fillna(0.0)
        if not bool(config.get("covariance_is_annualized", True)):
            cov = cov * annualization
    elif returns is not None and not returns.empty:
        return_frame = returns.copy().apply(pd.to_numeric, errors="coerce")
        configured_aliases = {**DEFAULT_ASSET_ALIASES, **dict(config.get("asset_aliases", {}))}
        rename: dict[Any, str] = {}
        for asset in assets:
            if asset in return_frame.columns:
                continue
            for candidate in configured_aliases.get(str(asset), (str(asset), str(asset).lower())):
                if candidate in return_frame.columns:
                    rename[candidate] = str(asset)
                    break
        return_frame = return_frame.rename(columns=rename)
        lookback = max(2, int(config["vol_lookback"]))
        sample = return_frame.reindex(columns=assets).tail(lookback)
        cov = sample.cov(min_periods=max(2, min(10, len(sample)))) * annualization
        cov = cov.reindex(index=assets, columns=assets)
    else:
        cov = pd.DataFrame(0.0, index=assets, columns=assets)

    cov = repair_covariance(
        cov,
        assets,
        fallback_variance=fallback**2,
        eigenvalue_floor=float(config.get("covariance_eigenvalue_floor", 1e-8)),
    )
    volatility = pd.Series(np.sqrt(np.maximum(np.diag(cov.to_numpy()), 0.0)), index=assets)
    volatility = volatility.clip(lower=min_vol).replace([np.inf, -np.inf], fallback).fillna(fallback)
    return cov, volatility


def _asset_class(asset: str, mapping: Mapping[str, str]) -> str:
    if asset in mapping:
        return str(mapping[asset])
    # Relative-value instruments inherit the economically dominant first leg.
    first_leg = str(asset).split("/", 1)[0]
    if first_leg in mapping:
        return str(mapping[first_leg])
    for code, aliases in DEFAULT_ASSET_ALIASES.items():
        if first_leg in aliases and code in mapping:
            return str(mapping[code])
    return "other"


def _cap_group_gross(weights: pd.Series, groups: pd.Series, limits: Mapping[str, float], default: float) -> pd.Series:
    result = weights.copy()
    for group, members in groups.groupby(groups):
        index = members.index
        gross = float(result.loc[index].abs().sum())
        limit = max(0.0, float(limits.get(str(group), default)))
        if gross > limit and gross > 0:
            result.loc[index] *= limit / gross
    return result


def _cap_commodity_gross(
    weights: pd.Series,
    asset_classes: pd.Series,
    commodity_classes: Sequence[str],
    limit: float,
) -> pd.Series:
    """Cap the combined grain-and-energy commodity family gross exposure."""

    result = weights.copy()
    members = asset_classes.index[asset_classes.isin({str(item) for item in commodity_classes})]
    gross = float(result.reindex(members).abs().sum())
    family_limit = max(0.0, float(limit))
    if gross > family_limit and gross > 0.0:
        result.loc[members] *= family_limit / gross
    return result


def _cap_net(weights: pd.Series, max_net: float) -> pd.Series:
    """Reduce only the over-represented side until the net limit is met."""

    result = weights.copy()
    net = float(result.sum())
    limit = max(0.0, float(max_net))
    if abs(net) <= limit:
        return result
    if net > limit:
        positive = result.clip(lower=0.0)
        positive_sum = float(positive.sum())
        reduction = net - limit
        if positive_sum > 0:
            result.loc[positive > 0] *= max(0.0, 1.0 - reduction / positive_sum)
    else:
        negative = -result.clip(upper=0.0)
        negative_sum = float(negative.sum())
        reduction = -net - limit
        if negative_sum > 0:
            result.loc[negative > 0] *= max(0.0, 1.0 - reduction / negative_sum)
    return result


def _factor_frame(
    assets: pd.Index,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None,
) -> pd.DataFrame | None:
    if factor_exposures is None:
        return None
    if isinstance(factor_exposures, Mapping):
        frame = pd.DataFrame.from_dict(factor_exposures, orient="index")
    elif isinstance(factor_exposures, pd.DataFrame):
        frame = factor_exposures.copy()
    else:
        raise TypeError("factor_exposures must be a DataFrame, mapping, or None")
    frame.index = frame.index.astype(str)
    rename: dict[str, str] = {}
    for asset in assets.astype(str):
        if asset in frame.index:
            continue
        aliases = DEFAULT_ASSET_ALIASES.get(asset, (asset, asset.lower()))
        match = next((candidate for candidate in aliases if candidate in frame.index), None)
        if match is not None:
            rename[str(match)] = asset
    frame = frame.rename(index=rename)
    return frame.apply(pd.to_numeric, errors="coerce").reindex(assets).fillna(0.0)


def _enforce_factor_limits(
    weights: pd.Series,
    exposures: pd.DataFrame | None,
    limits: Mapping[str, float] | float,
    iterations: int = 4,
) -> pd.Series:
    """Shrink same-direction contributors until factor exposure is in bounds.

    A risk constraint must not invent a hedge in an otherwise inactive asset.
    This implementation therefore scales only existing positions that
    contribute to the excessive side and never reverses their sign.
    """

    if exposures is None or exposures.empty:
        return weights
    result = weights.copy()
    for _ in range(max(1, iterations)):
        changed = False
        for factor in exposures.columns:
            vector = exposures[factor].astype(float)
            exposure = float(result.dot(vector))
            limit = float(limits.get(str(factor), limits.get("default", 0.35))) if isinstance(limits, Mapping) else float(limits)
            limit = max(0.0, limit)
            if abs(exposure) <= limit + 1e-12:
                continue
            contribution = result * vector
            same_direction = contribution * np.sign(exposure) > 0
            supporting_exposure = float((contribution.loc[same_direction] * np.sign(exposure)).sum())
            if supporting_exposure <= 0:
                continue
            required_reduction = abs(exposure) - limit
            scale = max(0.0, 1.0 - required_reduction / supporting_exposure)
            result.loc[same_direction] *= scale
            changed = True
        if not changed:
            break
    return result


def expected_portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Annualized volatility for weights and an annualized covariance matrix."""

    aligned = repair_covariance(covariance, weights.index)
    variance = float(weights.to_numpy() @ aligned.to_numpy() @ weights.to_numpy())
    return float(np.sqrt(max(0.0, variance)))


def construct_portfolio(
    signals: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Turn current signal records into volatility-targeted constrained weights.

    The default sequence is signal-strength/confidence weighting, inverse-vol
    sizing, target-vol scaling, then position, group, gross, net, and factor
    constraints.  Constraints are only allowed to reduce ex-ante risk.
    """

    cfg = _portfolio_config(config)
    signal, confidence = _latest_signal_vector(signals)
    signal.index = signal.index.astype(str)
    confidence = confidence.reindex(signal.index).fillna(0.0)
    signal = signal.where(signal.abs() >= float(cfg["min_signal"]), 0.0)
    confidence = confidence.where(confidence >= float(cfg["confidence_floor"]), 0.0)
    assets = signal.index
    if len(assets) == 0:
        return pd.Series(dtype=float, name="weight")

    tradability_exclusions: dict[str, str] = {}
    if returns is not None and not returns.empty:
        history = returns.copy().apply(pd.to_numeric, errors="coerce")
        alias_map = {**DEFAULT_ASSET_ALIASES, **dict(cfg.get("asset_aliases", {}))}
        renamed: dict[Any, str] = {}
        for asset in assets:
            if asset in history.columns:
                continue
            match = next(
                (
                    candidate
                    for candidate in alias_map.get(str(asset), (str(asset), str(asset).lower()))
                    if candidate in history.columns
                ),
                None,
            )
            if match is not None:
                renamed[match] = str(asset)
        history = history.rename(columns=renamed).reindex(columns=assets)
        minimum_observations = max(1, int(cfg.get("min_return_observations", 20)))
        require_latest = bool(cfg.get("require_latest_return", True))
        for asset in assets:
            observed = int(history[asset].notna().sum())
            if observed < minimum_observations:
                tradability_exclusions[str(asset)] = (
                    f"only {observed} valid returns; {minimum_observations} required"
                )
            elif require_latest and pd.isna(history[asset].iloc[-1]):
                tradability_exclusions[str(asset)] = "latest return/mark is unavailable"
        if tradability_exclusions:
            excluded = pd.Index(tradability_exclusions)
            signal.loc[signal.index.intersection(excluded)] = 0.0
            confidence.loc[confidence.index.intersection(excluded)] = 0.0

    cov, volatility = _covariance_and_volatility(assets, returns, covariance, cfg)
    score = signal * confidence
    raw = score / volatility
    gross_raw = float(raw.abs().sum())
    if gross_raw <= 0 or not np.isfinite(gross_raw):
        result = pd.Series(0.0, index=assets, name="weight")
        result.attrs["expected_volatility"] = 0.0
        result.attrs["tradability_exclusions"] = tradability_exclusions
        return result

    # Normalize before volatility targeting so the initial scale is stable
    # across changes in universe size.
    weights = raw / gross_raw
    target_vol = max(0.0, float(cfg["target_volatility"]))
    predicted = expected_portfolio_volatility(weights, cov)
    if predicted > 0 and target_vol > 0:
        weights *= target_vol / predicted

    max_position = max(0.0, float(cfg["max_position_weight"]))
    weights = weights.clip(-max_position, max_position)

    class_mapping = {**DEFAULT_ASSET_CLASSES, **dict(cfg.get("asset_classes", {}))}
    groups = pd.Series({_asset: _asset_class(_asset, class_mapping) for _asset in assets})
    default_class_limit = max(0.0, float(cfg["max_asset_class_exposure"]))
    class_limits = dict(cfg.get("asset_class_limits", {}))
    class_limits.setdefault("fertilizer_equity", float(cfg["max_fertilizer_equity_exposure"]))
    weights = _cap_group_gross(weights, groups, class_limits, default_class_limit)
    commodity_classes = tuple(cfg.get("commodity_asset_classes", ("grain", "energy")))
    weights = _cap_commodity_gross(
        weights,
        groups,
        commodity_classes,
        float(cfg["max_commodity_exposure"]),
    )

    exposures = _factor_frame(assets, factor_exposures)
    factor_limits = cfg.get("factor_limits", float(cfg["max_factor_exposure"]))
    weights = _enforce_factor_limits(weights, exposures, factor_limits)
    # Projection may move an individual instrument, so reapply hard limits.
    weights = weights.clip(-max_position, max_position)
    weights = _cap_group_gross(weights, groups, class_limits, default_class_limit)
    weights = _cap_commodity_gross(
        weights,
        groups,
        commodity_classes,
        float(cfg["max_commodity_exposure"]),
    )

    max_gross = max(0.0, float(cfg["max_gross_exposure"]))
    gross = float(weights.abs().sum())
    if gross > max_gross and gross > 0:
        weights *= max_gross / gross
    weights = _cap_net(weights, float(cfg["max_net_exposure"]))

    # Net/group caps can change the balance between offsetting factor legs.
    # A final uniform haircut guarantees every factor constraint without
    # weakening any already-satisfied position, group, gross, or net limit.
    if exposures is not None and not exposures.empty:
        factor_scale = 1.0
        final_factor_exposure = exposures.mul(weights, axis=0).sum()
        for factor, exposure in final_factor_exposure.items():
            limit = (
                float(factor_limits.get(str(factor), factor_limits.get("default", 0.35)))
                if isinstance(factor_limits, Mapping)
                else float(factor_limits)
            )
            if abs(float(exposure)) > max(0.0, limit) and abs(float(exposure)) > 0:
                factor_scale = min(factor_scale, max(0.0, limit) / abs(float(exposure)))
        if factor_scale < 1.0:
            weights *= factor_scale

    # Numerical projection can occasionally increase covariance risk even while
    # reducing gross.  Never finish above the requested target.
    final_vol = expected_portfolio_volatility(weights, cov)
    if target_vol > 0 and final_vol > target_vol:
        weights *= target_vol / final_vol
        final_vol = target_vol

    weights.name = "weight"
    weights.attrs.update(
        {
            "expected_volatility": float(final_vol),
            "gross_exposure": float(weights.abs().sum()),
            "net_exposure": float(weights.sum()),
            "asset_class_exposure": weights.groupby(groups).sum().to_dict(),
            "asset_class_gross": weights.abs().groupby(groups).sum().to_dict(),
            "commodity_gross_exposure": float(
                weights.loc[groups.index[groups.isin(set(commodity_classes))]].abs().sum()
            ),
            "factor_exposure": (
                exposures.mul(weights, axis=0).sum().to_dict() if exposures is not None else {}
            ),
            "covariance_repair": dict(cov.attrs.get("repair_diagnostics", {})),
            "tradability_exclusions": tradability_exclusions,
        }
    )
    return weights.sort_index()


def construct_portfolio_history(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    factor_exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply :func:`construct_portfolio` at every signal date using past data only."""

    if not {"date", "asset", "signal"}.issubset(signals.columns):
        raise ValueError("long-form signals require date, asset, and signal columns")
    cfg = _portfolio_config(config)
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return_frame = returns.copy().apply(pd.to_numeric, errors="coerce")
    return_frame.index = pd.to_datetime(return_frame.index)
    rows: list[pd.Series] = []
    dates: list[pd.Timestamp] = []
    lookback = max(2, int(cfg["vol_lookback"]))
    annualization = float(cfg["annualization"])
    # Compute the rolling covariance panel once in pandas' optimized rolling
    # engine.  Re-estimating a DataFrame covariance inside every date loop is
    # materially slower on multi-year daily histories.
    rolling_covariance = return_frame.rolling(lookback, min_periods=2).cov() * annualization
    for date, group in frame.sort_values("date").groupby("date", sort=True):
        try:
            covariance = rolling_covariance.xs(pd.Timestamp(date), level=0)
        except (KeyError, IndexError):
            covariance = None
        weights = construct_portfolio(
            group,
            returns=return_frame.loc[: pd.Timestamp(date)].tail(lookback),
            covariance=covariance,
            factor_exposures=factor_exposures,
            config=cfg,
        )
        rows.append(weights)
        dates.append(pd.Timestamp(date))
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date")).fillna(0.0)
    return result.sort_index()


# Short aliases used by the CLI/reporting layer.
build_portfolio = construct_portfolio
portfolio_weights = construct_portfolio


__all__ = [
    "aggregate_executable_signals",
    "DEFAULT_ASSET_CLASSES",
    "DEFAULT_ASSET_ALIASES",
    "DEFAULT_PORTFOLIO_CONFIG",
    "build_portfolio",
    "construct_portfolio",
    "construct_portfolio_history",
    "expected_portfolio_volatility",
    "portfolio_weights",
]
