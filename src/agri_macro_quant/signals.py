"""Interpretable directional and relative-value trading signals.

The functions in this module deliberately operate on pandas objects and plain
dictionaries.  They do not know where the data came from and they do not place
orders.  A signal observed at time ``t`` is intended to be executed no earlier
than ``t + 1`` by :mod:`agri_macro_quant.backtest`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "date",
    "asset",
    "signal_type",
    "signal",
    "direction",
    "strength",
    "confidence",
    "expected_horizon",
    "current_regime",
    "expected_next_regime",
    "economic_driver",
    "invalidation_condition",
]


DEFAULT_SIGNAL_SPECS: dict[str, dict[str, Any]] = {
    "ZW": {
        "factor_weights": {
            "weather_stress": 0.25,
            "crop_condition": -0.15,
            "yield_revision": -0.15,
            "exporter_stocks_to_use": -0.20,
            "policy_stress": 0.10,
            "wheat_momentum": 0.15,
            "physical_scarcity": 0.15,
        },
        "regime_weights": {
            "weather_shock": 0.25,
            "food_security": 0.20,
            "policy_stress": 0.10,
            "supply_response": -0.15,
        },
        "horizon": "2-6 weeks",
        "driver": "weather, exporter inventories, and physical scarcity",
        "invalidation": (
            "weather stress eases, yield revisions turn positive, or exporter "
            "stocks-to-use improves"
        ),
        "short_invalidation": (
            "weather stress intensifies, yield revisions turn negative, or exporter "
            "stocks-to-use tightens"
        ),
    },
    "ZC": {
        "factor_weights": {
            "weather_stress": 0.25,
            "crop_condition": -0.15,
            "yield_revision": -0.15,
            "exporter_stocks_to_use": -0.15,
            "policy_stress": 0.10,
            "corn_momentum": 0.15,
            "corn_urea": 0.05,
        },
        "regime_weights": {
            "weather_shock": 0.25,
            "food_security": 0.15,
            "supply_response": -0.15,
        },
        "horizon": "2-8 weeks",
        "driver": "crop stress, inventory tightness, and corn momentum",
        "invalidation": (
            "crop conditions recover, yield expectations rise, or inventories "
            "loosen"
        ),
        "short_invalidation": (
            "crop conditions deteriorate, yield expectations fall, or inventories tighten"
        ),
    },
    "ZS": {
        "factor_weights": {
            "weather_stress": 0.20,
            "crop_condition": -0.15,
            "yield_revision": -0.15,
            "exporter_stocks_to_use": -0.15,
            "policy_stress": 0.10,
            "soybean_momentum": 0.15,
            "soy_fertilizer": 0.10,
        },
        "regime_weights": {
            "weather_shock": 0.20,
            "food_security": 0.15,
            "supply_response": -0.15,
        },
        "horizon": "2-8 weeks",
        "driver": "oilseed supply expectations and exportable surplus",
        "invalidation": (
            "South American supply improves or export demand weakens materially"
        ),
        "short_invalidation": (
            "South American supply deteriorates or export demand strengthens materially"
        ),
    },
    "CF": {
        "factor_weights": {
            "corn_urea": 0.25,
            "ag_affordability": 0.15,
            "nitrogen_margin": 0.25,
            "farmer_margin": 0.15,
            "cf_momentum": 0.10,
            "natural_gas": -0.10,
        },
        "regime_weights": {
            "input_expansion": 0.30,
            "supply_response": 0.10,
            "monetary_tightening": -0.10,
        },
        "horizon": "1-3 months",
        "driver": "nitrogen margins and farmer input affordability",
        "invalidation": (
            "nitrogen margins compress or crop-to-fertilizer affordability falls"
        ),
        "short_invalidation": (
            "nitrogen margins expand or crop-to-fertilizer affordability rises"
        ),
    },
    "NTR": {
        "factor_weights": {
            "ag_affordability": 0.25,
            "farmer_margin": 0.20,
            "potash_margin": 0.20,
            "ntr_momentum": 0.15,
            "policy_stress": 0.05,
        },
        "regime_weights": {
            "input_expansion": 0.30,
            "supply_response": 0.10,
            "monetary_tightening": -0.10,
        },
        "horizon": "1-4 months",
        "driver": "farmer economics and fertilizer demand",
        "invalidation": "fertilizer affordability or producer margins deteriorate",
        "short_invalidation": "fertilizer affordability or producer margins improve",
    },
    "MOS": {
        "factor_weights": {
            "ag_affordability": 0.20,
            "farmer_margin": 0.15,
            "phosphate_margin": 0.25,
            "mos_momentum": 0.15,
        },
        "regime_weights": {
            "input_expansion": 0.30,
            "monetary_tightening": -0.10,
        },
        "horizon": "1-4 months",
        "driver": "phosphate margins and agricultural input demand",
        "invalidation": "phosphate margins compress or acreage incentives reverse",
        "short_invalidation": "phosphate margins expand or acreage incentives strengthen",
    },
    "DXY": {
        "factor_weights": {
            "us2y": 0.25,
            "real_yield": 0.20,
            "dxy_momentum": 0.20,
            "breakeven": -0.10,
            "growth": 0.10,
        },
        "regime_weights": {"monetary_tightening": 0.30},
        "horizon": "2-8 weeks",
        "driver": "relative monetary tightening and real yields",
        "invalidation": "policy-rate expectations or real yields turn lower",
        "short_invalidation": "policy-rate expectations or real yields turn higher",
    },
    "US2Y": {
        "factor_weights": {
            "food_inflation": 0.25,
            "breakeven": 0.20,
            "fed_repricing": 0.25,
            "us2y_momentum": 0.15,
        },
        "regime_weights": {
            "food_inflation": 0.15,
            "monetary_tightening": 0.30,
        },
        "horizon": "2-8 weeks",
        "driver": "food-inflation transmission and policy repricing",
        "invalidation": "inflation expectations or hawkish policy repricing reverses",
        "short_invalidation": (
            "inflation expectations or hawkish policy repricing intensifies"
        ),
    },
}


DEFAULT_RELATIVE_PAIRS: dict[str, dict[str, Any]] = {
    "ZW/CF": {"long": "ZW", "short": "CF", "driver": "grain versus input equities"},
    "ZC/CF": {"long": "ZC", "short": "CF", "driver": "corn versus nitrogen producer"},
    "CF/NG": {"long": "CF", "short": "NG", "driver": "nitrogen producer margin"},
    "CF/NTR": {"long": "CF", "short": "NTR", "driver": "nitrogen relative value"},
    "NTR/MOS": {"long": "NTR", "short": "MOS", "driver": "potash versus phosphate"},
    "ZW/ZC": {"long": "ZW", "short": "ZC", "driver": "wheat relative scarcity"},
    "ZW/DXY": {"long": "ZW", "short": "DXY", "driver": "physical scarcity versus USD pressure"},
}


REGIME_ALIASES: dict[str, str] = {
    "weather_crop_supply_shock": "weather_shock",
    "weather_supply_shock": "weather_shock",
    "inventory_food_security_stress": "food_security",
    "food_security_stress": "food_security",
    "government_trade_policy_amplification": "policy_stress",
    "government_intervention": "policy_stress",
    "agricultural_input_expansion": "input_expansion",
    "fertilizer_expansion": "input_expansion",
    "food_inflation_transmission": "food_inflation",
    "monetary_tightening_stagflation": "monetary_tightening",
    "supply_response_mean_reversion": "supply_response",
}


def _canonical(value: Any) -> str:
    """Return a forgiving identifier used for column/spec matching."""

    token = "_".join(str(value).strip().lower().replace("/", "_").replace("-", "_").split())
    return REGIME_ALIASES.get(token, token)


def _column_lookup(frame: pd.DataFrame) -> dict[str, Any]:
    return {_canonical(column): column for column in frame.columns}


def _ensure_datetime_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    if "date" in result.columns and not isinstance(result.index, pd.DatetimeIndex):
        result = result.set_index("date")
    try:
        result.index = pd.to_datetime(result.index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have a datetime-like index") from exc
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    return result.apply(pd.to_numeric, errors="coerce")


def _causal_winsorize(frame: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Clip without using future observations."""

    if limit <= 0:
        return frame
    expanding_mean = frame.expanding(min_periods=2).mean()
    expanding_std = frame.expanding(min_periods=2).std(ddof=0).replace(0.0, np.nan)
    lower = expanding_mean - limit * expanding_std
    upper = expanding_mean + limit * expanding_std
    return frame.clip(lower=lower, upper=upper, axis=0)


def _dominant_regime(regime_scores: pd.DataFrame) -> pd.Series:
    if regime_scores.empty:
        return pd.Series("unknown", index=regime_scores.index, dtype="object")
    available = regime_scores.notna().any(axis=1)
    result = regime_scores.fillna(-np.inf).idxmax(axis=1).astype(str)
    return result.where(available, "unknown")


def _next_regime_series(
    index: pd.DatetimeIndex,
    current: pd.Series,
    transition_probs: Any,
) -> pd.Series:
    if transition_probs is None:
        return current.copy()
    # Direct integration with transitions.TransitionResult without importing
    # it here (which keeps the signal module independently reusable).
    if hasattr(transition_probs, "forecast_history"):
        histories = transition_probs.forecast_history
        if isinstance(histories, Mapping) and histories:
            transition_probs = histories.get("1m", histories.get("1_month", next(iter(histories.values()))))
    elif hasattr(transition_probs, "latest_probabilities"):
        transition_probs = transition_probs.latest_probabilities
    if isinstance(transition_probs, pd.Series):
        probabilities = pd.to_numeric(transition_probs, errors="coerce").dropna()
        choice = str(probabilities.idxmax()) if not probabilities.empty else "unknown"
        return pd.Series(choice, index=index, dtype="object")
    if isinstance(transition_probs, pd.DataFrame):
        if {"next_regime", "probability"}.issubset(transition_probs.columns):
            latest = transition_probs.copy()
            if "horizon" in latest.columns:
                preferred = latest.loc[latest["horizon"].astype(str).isin(["1m", "1_month"])]
                if not preferred.empty:
                    latest = preferred
            probability = pd.to_numeric(latest["probability"], errors="coerce")
            choice = str(latest.loc[probability.idxmax(), "next_regime"]) if probability.notna().any() else "unknown"
            return pd.Series(choice, index=index, dtype="object")
        if not isinstance(transition_probs.index, pd.DatetimeIndex):
            row = transition_probs.loc["1m"] if "1m" in transition_probs.index else transition_probs.iloc[-1]
            probability = pd.to_numeric(row, errors="coerce").dropna()
            choice = str(probability.idxmax()) if not probability.empty else "unknown"
            return pd.Series(choice, index=index, dtype="object")
        transitions = _ensure_datetime_index(transition_probs, "transition_probs")
        transitions = transitions.reindex(index, method="ffill")
        return _dominant_regime(transitions)
    if isinstance(transition_probs, Mapping):
        if transition_probs and all(not isinstance(value, Mapping) for value in transition_probs.values()):
            probabilities = pd.to_numeric(pd.Series(transition_probs), errors="coerce").dropna()
            choice = str(probabilities.idxmax()) if not probabilities.empty else "unknown"
            return pd.Series(choice, index=index, dtype="object")
        def choose(regime: str) -> str:
            row = transition_probs.get(regime, transition_probs.get(_canonical(regime), {}))
            if not row:
                return regime
            return str(max(row, key=row.get))

        return current.map(choose)
    raise TypeError("transition_probs must be a DataFrame, mapping, or None")


def _score_spec(
    factors: pd.DataFrame,
    regimes: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return score, confidence, and largest-contribution driver."""

    contributions: list[pd.Series] = []
    labels: list[str] = []
    factor_lookup = _column_lookup(factors)
    regime_lookup = _column_lookup(regimes)

    for requested, weight in dict(spec.get("factor_weights", {})).items():
        column = factor_lookup.get(_canonical(requested))
        if column is not None:
            contributions.append(factors[column].astype(float) * float(weight))
            labels.append(str(requested))

    for requested, weight in dict(spec.get("regime_weights", {})).items():
        column = regime_lookup.get(_canonical(requested))
        if column is not None:
            values = regimes[column].astype(float)
            # Regime inputs are commonly probabilities, percentages, or 0-100 scores.
            finite = values[np.isfinite(values)]
            scale = 100.0 if (not finite.empty and finite.abs().quantile(0.95) > 1.5) else 1.0
            centered = 2.0 * (values / scale).clip(0.0, 1.0) - 1.0
            contributions.append(centered * float(weight))
            labels.append(str(requested))

    if not contributions:
        empty = pd.Series(0.0, index=factors.index)
        return empty, empty, pd.Series(str(spec.get("driver", "insufficient data")), index=factors.index)

    matrix = pd.concat(contributions, axis=1)
    matrix.columns = labels
    configured_weight = sum(abs(float(x)) for x in dict(spec.get("factor_weights", {})).values())
    configured_weight += sum(abs(float(x)) for x in dict(spec.get("regime_weights", {})).values())
    used_weight = sum(
        abs(float(weight))
        for requested, weight in dict(spec.get("factor_weights", {})).items()
        if _canonical(requested) in factor_lookup
    ) + sum(
        abs(float(weight))
        for requested, weight in dict(spec.get("regime_weights", {})).items()
        if _canonical(requested) in regime_lookup
    )
    coverage = min(1.0, used_weight / configured_weight) if configured_weight else 0.0

    raw = matrix.sum(axis=1, min_count=1).fillna(0.0)
    score = np.tanh(raw).clip(-1.0, 1.0)
    signs = np.sign(matrix)
    directional_agreement = signs.mul(np.sign(raw), axis=0).gt(0).sum(axis=1)
    directional_count = matrix.notna().sum(axis=1).replace(0, np.nan)
    agreement = (directional_agreement / directional_count).fillna(0.0)
    breadth = (matrix.notna().sum(axis=1) / max(1, matrix.shape[1])).clip(0.0, 1.0)
    # Confidence rewards data coverage and agreement but remains conservative.
    confidence = 100.0 * (0.50 * coverage * breadth + 0.35 * agreement + 0.15 * score.abs())
    confidence = confidence.clip(0.0, 100.0)

    absolute = matrix.abs().fillna(-np.inf)
    driver = absolute.idxmax(axis=1).where(matrix.notna().any(axis=1), str(spec.get("driver", "unknown")))
    return score, confidence, driver.astype(str)


def _unit_setting(value: Any, asset: str, default: float) -> float:
    """Read a scalar/percent setting, optionally by asset, and bound to 0-1."""

    if hasattr(value, "score"):
        value = value.score
    if isinstance(value, Mapping):
        value = value.get(asset, value.get(str(asset).lower(), value.get("default", value.get("score", default))))
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if abs(number) > 1.5:
        number /= 100.0
    return float(np.clip(number, 0.0, 1.0))


def generate_trade_signals(
    factors: pd.DataFrame,
    regime_scores: pd.DataFrame | Any,
    transition_probs: pd.DataFrame | Mapping[str, Mapping[str, float]] | Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate transparent absolute-direction trade candidates.

    Parameters
    ----------
    factors:
        Date-indexed factor values.  Factors should normally already be
        standardized; the final weighted score is always compressed to
        ``[-1, 1]`` regardless of their scale.
    regime_scores:
        Date-indexed regime scores expressed as probabilities, 0-1 values, or
        0-100 values.
    transition_probs:
        Optional date-indexed next-regime probabilities, or a mapping from a
        current regime to ``{next_regime: probability}``.
    config:
        Optional overrides.  ``signal_specs`` (or ``assets``) replaces the
        default asset specifications.  ``min_signal`` controls the flat band.

    Returns
    -------
    pandas.DataFrame
        One canonical signal record per date and asset.  ``confidence`` is a
        percentage while ``signal`` and ``strength`` are unit scaled.
    """

    cfg = dict(config or {})
    factor_frame = _ensure_datetime_index(factors, "factors")
    if hasattr(regime_scores, "scores"):
        regime_input = regime_scores.scores
    elif hasattr(regime_scores, "probabilities"):
        regime_input = regime_scores.probabilities
    else:
        regime_input = regime_scores
    regime_frame = _ensure_datetime_index(regime_input, "regime_scores")
    index = factor_frame.index.union(regime_frame.index).sort_values()
    factor_frame = factor_frame.reindex(index).ffill()
    regime_frame = regime_frame.reindex(index).ffill()
    factor_frame = _causal_winsorize(factor_frame, float(cfg.get("winsorize", 6.0)))

    specs = cfg.get("signal_specs", cfg.get("assets", DEFAULT_SIGNAL_SPECS))
    if not isinstance(specs, Mapping):
        raise TypeError("config['signal_specs'] must be a mapping")

    current_regime = _dominant_regime(regime_frame)
    expected_next = _next_regime_series(index, current_regime, transition_probs)
    flat_band = max(0.0, float(cfg.get("min_signal", 0.05)))
    rows: list[pd.DataFrame] = []

    for asset, raw_spec in specs.items():
        spec = dict(raw_spec or {})
        score, confidence, driver = _score_spec(factor_frame, regime_frame, spec)
        quality = _unit_setting(
            spec.get("data_quality", cfg.get("data_quality", cfg.get("data_quality_score", 1.0))),
            str(asset),
            1.0,
        )
        model_agreement = _unit_setting(spec.get("model_agreement", cfg.get("model_agreement", 1.0)), str(asset), 1.0)
        reliability = _unit_setting(
            spec.get("historical_reliability", cfg.get("historical_reliability", 1.0)),
            str(asset),
            1.0,
        )
        crowding = _unit_setting(spec.get("crowding_risk", cfg.get("crowding_risk", 0.0)), str(asset), 0.0)
        event_risk = _unit_setting(spec.get("event_risk", cfg.get("event_risk", 0.0)), str(asset), 0.0)
        confidence = (
            confidence * quality * (0.75 + 0.25 * model_agreement) * (0.75 + 0.25 * reliability)
            - 15.0 * crowding
            - 10.0 * event_risk
        ).clip(0.0, 100.0)
        direction = pd.Series(
            np.select(
                [score > flat_band, score < -flat_band],
                ["long", "short"],
                default="flat",
            ),
            index=index,
        )
        long_invalidation = str(
            spec.get(
                "long_invalidation",
                spec.get(
                    "invalidation",
                    "the weighted signal falls through the flat threshold or the expected regime fails",
                ),
            )
        )
        short_invalidation = str(
            spec.get(
                "short_invalidation",
                "the weighted signal rises through the flat threshold or its bearish drivers reverse",
            )
        )
        flat_invalidation = str(
            spec.get(
                "flat_invalidation",
                "the weighted signal moves outside the flat threshold",
            )
        )
        invalidation = np.select(
            [direction.eq("long"), direction.eq("short")],
            [long_invalidation, short_invalidation],
            default=flat_invalidation,
        )
        frame = pd.DataFrame(
            {
                "date": index,
                "asset": str(asset),
                "signal_type": "absolute",
                "signal": score.to_numpy(),
                "direction": direction.to_numpy(),
                "strength": score.abs().to_numpy(),
                "confidence": confidence.to_numpy(),
                "expected_horizon": str(spec.get("horizon", cfg.get("default_horizon", "2-8 weeks"))),
                "current_regime": current_regime.to_numpy(),
                "expected_next_regime": expected_next.to_numpy(),
                "economic_driver": driver.to_numpy(),
                "invalidation_condition": invalidation,
            }
        )
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    result = pd.concat(rows, ignore_index=True)
    result["signal"] = pd.to_numeric(result["signal"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    result["strength"] = result["signal"].abs()
    result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce").fillna(0.0).clip(0.0, 100.0)
    return result[SIGNAL_COLUMNS].sort_values(["date", "asset"], ignore_index=True)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _pair_definitions(
    pairs: Mapping[str, Any] | Sequence[tuple[str, str]] | None,
) -> dict[str, dict[str, Any]]:
    if pairs is None:
        return {key: dict(value) for key, value in DEFAULT_RELATIVE_PAIRS.items()}
    if isinstance(pairs, Mapping):
        parsed: dict[str, dict[str, Any]] = {}
        for name, value in pairs.items():
            if isinstance(value, Mapping):
                parsed[str(name)] = dict(value)
            elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= 2:
                parsed[str(name)] = {"long": value[0], "short": value[1]}
            else:
                raise TypeError(f"invalid pair definition for {name!r}")
        return parsed
    parsed = {}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each pair must have exactly two assets")
        parsed[f"{pair[0]}/{pair[1]}"] = {"long": pair[0], "short": pair[1]}
    return parsed


def generate_relative_value_signals(
    prices: pd.DataFrame,
    pairs: Mapping[str, Any] | Sequence[tuple[str, str]] | None = None,
    regime_scores: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate price-ratio, return-spread, and beta-neutral pair signals.

    A positive signal means long the pair's ``long_leg`` and short its
    ``short_leg``.  Pair signals default to continuation because the economic
    use case is isolating an emerging causal divergence.  Set
    ``config={'style': 'mean_reversion'}`` to reverse the sign.
    """

    cfg = dict(config or {})
    frame = _ensure_datetime_index(prices, "prices")
    if (frame <= 0).any().any():
        # Non-positive observations cannot form a log ratio, but should not
        # invalidate other pairs/dates.
        frame = frame.where(frame > 0)
    definitions = _pair_definitions(pairs)
    methods = cfg.get("methods", ["price_ratio", "return_spread", "beta_neutral"])
    if isinstance(methods, str):
        methods = [methods]
    supported = {"price_ratio", "return_spread", "beta_neutral"}
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(f"unsupported relative-value methods: {sorted(unknown)}")

    window = max(5, int(cfg.get("lookback", 60)))
    min_periods = max(3, min(window, int(cfg.get("min_periods", max(10, window // 3)))))
    style_sign = -1.0 if str(cfg.get("style", "continuation")).lower() in {"mean_reversion", "reversion"} else 1.0
    flat_band = max(0.0, float(cfg.get("min_signal", 0.05)))
    returns = np.log(frame).diff()

    if regime_scores is None:
        current = pd.Series("unknown", index=frame.index)
    else:
        if hasattr(regime_scores, "scores"):
            regime_input = regime_scores.scores
        elif hasattr(regime_scores, "probabilities"):
            regime_input = regime_scores.probabilities
        else:
            regime_input = regime_scores
        regimes = _ensure_datetime_index(regime_input, "regime_scores").reindex(frame.index).ffill()
        current = _dominant_regime(regimes)

    records: list[pd.DataFrame] = []
    for name, spec in definitions.items():
        long_leg = str(spec.get("long", spec.get("numerator", "")))
        short_leg = str(spec.get("short", spec.get("denominator", "")))
        if long_leg not in frame.columns or short_leg not in frame.columns:
            continue

        long_return = returns[long_leg]
        short_return = returns[short_leg]
        short_variance = short_return.rolling(window, min_periods=min_periods).var(ddof=0)
        beta = long_return.rolling(window, min_periods=min_periods).cov(short_return).div(short_variance.replace(0.0, np.nan))
        beta = beta.clip(float(cfg.get("min_beta", -3.0)), float(cfg.get("max_beta", 3.0))).fillna(1.0)

        values: dict[str, pd.Series] = {
            "price_ratio": _rolling_zscore(np.log(frame[long_leg] / frame[short_leg]), window, min_periods),
            "return_spread": _rolling_zscore(long_return - short_return, window, min_periods),
            "beta_neutral": _rolling_zscore(long_return - beta * short_return, window, min_periods),
        }
        for method in methods:
            raw = values[method]
            signal = pd.Series(np.tanh(style_sign * raw), index=frame.index).fillna(0.0).clip(-1.0, 1.0)
            data_available = raw.notna().astype(float)
            sample_fraction = (
                long_return.notna().rolling(window, min_periods=1).sum() / float(window)
            ).clip(0.0, 1.0)
            confidence = 100.0 * data_available * (
                0.55 * sample_fraction + 0.30 * np.minimum(1.0, raw.abs() / 2.0).fillna(0.0) + 0.15
            )
            quality = _unit_setting(cfg.get("data_quality", cfg.get("data_quality_score", 1.0)), str(name), 1.0)
            reliability = _unit_setting(cfg.get("historical_reliability", 1.0), str(name), 1.0)
            crowding = _unit_setting(cfg.get("crowding_risk", 0.0), str(name), 0.0)
            event_risk = _unit_setting(cfg.get("event_risk", 0.0), str(name), 0.0)
            confidence = (
                confidence * quality * (0.75 + 0.25 * reliability)
                - 15.0 * crowding
                - 10.0 * event_risk
            ).clip(0.0, 100.0)
            direction = np.select(
                [signal > flat_band, signal < -flat_band],
                ["long_spread", "short_spread"],
                default="flat",
            )
            economic_driver = str(spec.get("driver", f"{long_leg} relative to {short_leg}"))
            invalidation = str(
                spec.get(
                    "invalidation",
                    f"the {method.replace('_', ' ')} divergence reverses or loses statistical significance",
                )
            )
            candidate = pd.DataFrame(
                {
                    "date": frame.index,
                    "asset": str(name),
                    "signal_type": f"relative_value:{method}",
                    "signal": signal.to_numpy(),
                    "direction": direction,
                    "strength": signal.abs().to_numpy(),
                    "confidence": confidence.clip(0.0, 100.0).to_numpy(),
                    "expected_horizon": str(spec.get("horizon", cfg.get("default_horizon", "2-8 weeks"))),
                    "current_regime": current.to_numpy(),
                    "expected_next_regime": current.to_numpy(),
                    "economic_driver": economic_driver,
                    "invalidation_condition": invalidation,
                    "long_leg": long_leg,
                    "short_leg": short_leg,
                    "hedge_ratio": beta.to_numpy() if method == "beta_neutral" else 1.0,
                    "spread_zscore": raw.to_numpy(),
                }
            )
            records.append(candidate)

    extra = ["long_leg", "short_leg", "hedge_ratio", "spread_zscore"]
    if not records:
        return pd.DataFrame(columns=SIGNAL_COLUMNS + extra)
    result = pd.concat(records, ignore_index=True)
    return result[SIGNAL_COLUMNS + extra].sort_values(
        ["date", "asset", "signal_type"], ignore_index=True
    )


# Concise aliases used by notebooks and integrations.
generate_signals = generate_trade_signals
generate_trade_candidates = generate_trade_signals
relative_value_signals = generate_relative_value_signals


__all__ = [
    "DEFAULT_RELATIVE_PAIRS",
    "DEFAULT_SIGNAL_SPECS",
    "SIGNAL_COLUMNS",
    "generate_relative_value_signals",
    "generate_signals",
    "generate_trade_candidates",
    "generate_trade_signals",
    "relative_value_signals",
]
