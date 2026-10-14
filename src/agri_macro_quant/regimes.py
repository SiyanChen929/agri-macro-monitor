"""Transparent and statistical seven-regime estimation.

The transparent model is the production baseline: each score is a decomposable
logistic transform of economically signed MVP factors.  The optional
statistical estimator uses expanding, walk-forward multinomial logistic models
and falls back to transparent probabilities when training data are inadequate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .factors import MVP_FACTOR_COLUMNS


REGIME_NAMES: tuple[str, ...] = (
    "weather_crop_supply_shock",
    "inventory_food_security_stress",
    "government_trade_policy_amplification",
    "agricultural_input_expansion",
    "food_inflation_transmission",
    "monetary_tightening_stagflation",
    "supply_response_mean_reversion",
)

REGIME_LABELS: dict[str, str] = {
    "weather_crop_supply_shock": "Weather / Crop Supply Shock",
    "inventory_food_security_stress": "Inventory / Food Security Stress",
    "government_trade_policy_amplification": "Government / Trade Policy Amplification",
    "agricultural_input_expansion": "Agricultural Input Expansion",
    "food_inflation_transmission": "Food Inflation Transmission",
    "monetary_tightening_stagflation": "Monetary Tightening / Stagflation",
    "supply_response_mean_reversion": "Supply Response / Mean Reversion",
}


# Signs preserve the economic meaning of the underlying factor.  For example,
# strong crop condition and high exporter inventories reduce supply stress.
DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "weather_crop_supply_shock": {
        "crop_condition": -0.90,
        "yield_revision": -1.00,
        "weather_stress": 1.20,
        "wheat_momentum": 0.25,
    },
    "inventory_food_security_stress": {
        "exporter_stocks_to_use": -1.30,
        "policy_stress": 0.35,
        "weather_stress": 0.25,
        "wheat_momentum": 0.65,
    },
    "government_trade_policy_amplification": {
        "exporter_stocks_to_use": -0.40,
        "policy_stress": 1.40,
        "weather_stress": 0.25,
        "wheat_momentum": 0.30,
    },
    "agricultural_input_expansion": {
        "crop_condition": 0.20,
        "yield_revision": 0.20,
        "corn_urea": 1.20,
        "us2y": -0.15,
        "wheat_momentum": 0.45,
    },
    "food_inflation_transmission": {
        "exporter_stocks_to_use": -0.25,
        "policy_stress": 0.25,
        "weather_stress": 0.20,
        "breakeven": 0.80,
        "wheat_momentum": 0.90,
    },
    "monetary_tightening_stagflation": {
        "crop_condition": -0.20,
        "dxy": 0.25,
        "us2y": 0.95,
        "breakeven": 0.95,
        "wheat_momentum": 0.35,
    },
    "supply_response_mean_reversion": {
        "crop_condition": 0.65,
        "yield_revision": 0.85,
        "exporter_stocks_to_use": 0.55,
        "policy_stress": -0.30,
        "weather_stress": -0.80,
        "corn_urea": 0.65,
        "wheat_momentum": -0.40,
    },
}

DEFAULT_REGIME_CONFIG: dict[str, Any] = {
    "weights": DEFAULT_REGIME_WEIGHTS,
    "intercepts": {},
    "temperature": 1.0,
    "min_coverage": 0.35,
    "rescale_missing": True,
    "raw_clip": 12.0,
}


_REGIME_ALIASES: dict[str, str] = {
    "weather_supply_shock": "weather_crop_supply_shock",
    "weather_shock": "weather_crop_supply_shock",
    "food_security": "inventory_food_security_stress",
    "inventory_stress": "inventory_food_security_stress",
    "policy_amplification": "government_trade_policy_amplification",
    "policy_stress": "government_trade_policy_amplification",
    "input_expansion": "agricultural_input_expansion",
    "food_inflation": "food_inflation_transmission",
    "monetary_tightening": "monetary_tightening_stagflation",
    "stagflation": "monetary_tightening_stagflation",
    "supply_response": "supply_response_mean_reversion",
    "mean_reversion": "supply_response_mean_reversion",
}


_FACTOR_ALIASES: dict[str, str] = {
    "crop_condition_z": "crop_condition",
    "yield_revision_z": "yield_revision",
    "exporter_stocks_to_use_z": "exporter_stocks_to_use",
    "policy_stress_z": "policy_stress",
    "weather_stress_z": "weather_stress",
    "corn_urea_z": "corn_urea",
    "dxy_z": "dxy",
    "us_2y": "us2y",
    "us_2y_z": "us2y",
    "us2y_z": "us2y",
    "breakeven_5y": "breakeven",
    "breakeven_5y_z": "breakeven",
    "breakeven_z": "breakeven",
    "wheat_momentum_20_z": "wheat_momentum",
    "wheat_momentum_60_z": "wheat_momentum",
    "wheat_momentum_z": "wheat_momentum",
}


def regime_config_from_mapping(
    payload: Mapping[str, Any],
    *,
    mvp_only: bool = True,
) -> dict[str, Any]:
    """Convert loaded regime YAML (or an equivalent mapping) for scoring.

    The shipped YAML uses top-level ``weights``/``intercepts`` mappings.  The
    legacy ``{"model": ..., "regimes": ...}`` shape is also supported.  Short
    regime aliases and familiar ``*_z`` MVP factor names are canonicalised.

    Configuration is deliberately fail-closed: unknown fields, regimes, or
    factors raise immediately instead of being skipped.  ``mvp_only`` remains
    in the public signature for compatibility, but this scorer always accepts
    only the ten factors in :data:`MVP_FACTOR_COLUMNS`.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("regime configuration payload must be a mapping")
    del mvp_only  # retained only as a backwards-compatible keyword argument

    allowed_top_level = {
        "model",
        "regimes",
        "weights",
        "intercepts",
        "labels",
        "temperature",
        "min_coverage",
        "rescale_missing",
        "raw_clip",
    }
    unknown_top_level = sorted(set(map(str, payload)).difference(allowed_top_level))
    if unknown_top_level:
        raise ValueError(
            "regime configuration contains unknown top-level key(s): "
            f"{unknown_top_level}"
        )
    if "regimes" in payload and "weights" in payload:
        raise ValueError("regime configuration cannot define both `regimes` and `weights`")

    raw_model = payload.get("model") or {}
    if not isinstance(raw_model, Mapping):
        raise ValueError("regime configuration `model` must be a mapping")
    model = dict(raw_model)
    allowed_model = {"temperature", "min_coverage", "rescale_missing", "raw_clip"}
    unknown_model = sorted(set(map(str, model)).difference(allowed_model))
    if unknown_model:
        raise ValueError(
            "regime configuration model contains unknown key(s): "
            f"{unknown_model}"
        )

    def canonical_regime_name(supplied: Any, context: str) -> str:
        raw_name = str(supplied)
        name = _REGIME_ALIASES.get(raw_name, raw_name)
        if name not in REGIME_NAMES:
            raise ValueError(f"{context} contains unknown regime {raw_name!r}")
        return name

    def canonical_named_mapping(raw: Any, context: str) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context} must be a mapping")
        output: dict[str, Any] = {}
        for supplied_name, value in raw.items():
            name = canonical_regime_name(supplied_name, context)
            if name in output:
                raise ValueError(
                    f"{context} defines regime {name!r} more than once via aliases"
                )
            output[name] = value
        return output

    top_intercepts = canonical_named_mapping(
        payload.get("intercepts"), "regime configuration intercepts"
    )
    top_labels = canonical_named_mapping(
        payload.get("labels"), "regime configuration labels"
    )

    raw_definitions = payload.get("regimes", payload.get("weights", {})) or {}
    if not isinstance(raw_definitions, Mapping):
        key = "regimes" if "regimes" in payload else "weights"
        raise ValueError(f"regime configuration `{key}` must be a mapping")
    nested_shape = "regimes" in payload

    weights_output: dict[str, dict[str, float]] = {}
    intercepts: dict[str, float] = {}
    labels: dict[str, str] = {}
    seen_regimes: set[str] = set()
    for supplied_name, raw_definition in raw_definitions.items():
        name = canonical_regime_name(supplied_name, "regime configuration")
        if name in seen_regimes:
            raise ValueError(
                f"regime configuration defines regime {name!r} more than once via aliases"
            )
        seen_regimes.add(name)
        if nested_shape:
            if not isinstance(raw_definition, Mapping):
                raise ValueError(f"regime definition {supplied_name!r} must be a mapping")
            definition = dict(raw_definition)
            unknown_definition = sorted(
                set(map(str, definition)).difference({"weights", "intercept", "label"})
            )
            if unknown_definition:
                raise ValueError(
                    f"regime definition {supplied_name!r} contains unknown key(s): "
                    f"{unknown_definition}"
                )
            if "weights" not in definition:
                raise ValueError(
                    f"regime definition {supplied_name!r} must contain `weights`"
                )
            raw_weights = definition.get("weights") or {}
            intercept = definition.get("intercept", top_intercepts.get(name, 0.0))
            label = definition.get(
                "label", top_labels.get(name, REGIME_LABELS.get(name, name))
            )
        else:
            raw_weights = raw_definition
            intercept = top_intercepts.get(name, 0.0)
            label = top_labels.get(name, REGIME_LABELS.get(name, name))
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"weights for regime {supplied_name!r} must be a mapping")
        if not raw_weights:
            raise ValueError(f"weights for regime {supplied_name!r} cannot be empty")
        translated: dict[str, float] = {}
        for supplied_factor, raw_weight in dict(raw_weights or {}).items():
            factor = _FACTOR_ALIASES.get(str(supplied_factor), str(supplied_factor))
            if factor not in MVP_FACTOR_COLUMNS:
                raise ValueError(
                    f"weights for regime {supplied_name!r} contain unknown factor "
                    f"{str(supplied_factor)!r}"
                )
            if factor in translated:
                raise ValueError(
                    f"weights for regime {supplied_name!r} define factor {factor!r} "
                    "more than once via aliases"
                )
            translated[factor] = float(raw_weight)
        weights_output[name] = translated
        intercepts[name] = float(intercept)
        labels[name] = str(label)

    # Top-level intercepts and labels may customize regimes whose default
    # weights are retained because they were omitted from a partial override.
    for name, value in top_intercepts.items():
        intercepts.setdefault(name, float(value))
    for name, value in top_labels.items():
        labels.setdefault(name, str(value))

    result: dict[str, Any] = {
        "weights": weights_output,
        "intercepts": intercepts,
        "labels": labels,
        "ignored_weights": {},
    }
    for key in ("temperature", "min_coverage", "rescale_missing", "raw_clip"):
        if key in model:
            result[key] = model[key]
    for key in ("temperature", "min_coverage", "rescale_missing", "raw_clip"):
        if key in payload:
            result[key] = payload[key]
    return result


@dataclass(frozen=True)
class RegimeResult:
    """Outputs of the transparent seven-regime model."""

    scores: pd.DataFrame
    raw_scores: pd.DataFrame
    contributions: pd.DataFrame
    coverage: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_contributions(self) -> pd.DataFrame:
        """Contribution table for the most recent score date."""

        if self.contributions.empty:
            return self.contributions.copy()
        date = self.scores.index[-1]
        latest = self.contributions.loc[self.contributions["date"] == date].copy()
        return latest.sort_values(
            ["regime", "absolute_raw_contribution"], ascending=[True, False]
        ).reset_index(drop=True)

    @property
    def dominant_regime(self) -> pd.Series:
        """Highest-scoring available regime at each date."""

        return _safe_idxmax(self.scores)


@dataclass(frozen=True)
class StatisticalRegimeResult:
    """Walk-forward statistical probabilities and final model diagnostics."""

    probabilities: pd.DataFrame
    coefficients: pd.DataFrame
    intercepts: pd.Series
    method: str
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_probabilities(self) -> pd.Series:
        if self.probabilities.empty:
            return pd.Series(dtype=float, name="probability")
        result = self.probabilities.iloc[-1].sort_values(ascending=False)
        result.name = "probability"
        return result

    @property
    def dominant_regime(self) -> pd.Series:
        return _safe_idxmax(self.probabilities)


def _safe_idxmax(frame: pd.DataFrame) -> pd.Series:
    """Row-wise argmax that remains defined as pandas tightens all-NA rules."""

    available = frame.notna().any(axis=1)
    result = frame.fillna(-np.inf).idxmax(axis=1)
    return result.where(available)


def _canonical_factor_frame(factors: pd.DataFrame) -> pd.DataFrame:
    token_map = {
        "".join(c for c in str(column).lower() if c.isalnum()): column
        for column in factors.columns
    }
    aliases = {
        "crop_condition": ("cropcondition", "f1", "f1cropcondition"),
        "yield_revision": ("yieldrevision", "f2", "f2yieldrevision"),
        "exporter_stocks_to_use": (
            "exporterstockstouse",
            "exporterstu",
            "f3",
            "f3exporterstockstouse",
        ),
        "policy_stress": ("policystress", "f4", "f4policystress"),
        "weather_stress": ("weatherstress", "f5", "f5weatherstress"),
        "corn_urea": ("cornurea", "cornurearatio", "f6", "f6cornurea"),
        "dxy": ("dxy", "f7", "f7dxy"),
        "us2y": ("us2y", "us2yyield", "f8", "f8us2y"),
        "breakeven": ("breakeven", "f9", "f9breakeven"),
        "wheat_momentum": (
            "wheatmomentum",
            "wheatmom",
            "f10",
            "f10wheatmomentum",
        ),
    }
    output: dict[str, pd.Series] = {}
    for name in MVP_FACTOR_COLUMNS:
        if name in factors:
            column = name
        else:
            column = next(
                (token_map[token] for token in aliases[name] if token in token_map), None
            )
        output[name] = (
            pd.to_numeric(factors[column], errors="coerce").astype(float)
            if column is not None
            else pd.Series(np.nan, index=factors.index, dtype=float)
        )
    return pd.DataFrame(output, index=factors.index)


def _sigmoid(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _regime_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_REGIME_CONFIG)
    merged["weights"] = {
        regime: dict(weights) for regime, weights in DEFAULT_REGIME_WEIGHTS.items()
    }
    merged["intercepts"] = {}
    if config:
        update = dict(config)
        supplied_weights = update.pop("weights", update.pop("regime_weights", None))
        merged.update(update)
        if supplied_weights is not None:
            # A supplied regime definition replaces that regime, while omitted
            # regimes retain transparent defaults.
            for regime, weights in dict(supplied_weights).items():
                merged["weights"][str(regime)] = {
                    str(factor): float(weight)
                    for factor, weight in dict(weights).items()
                }
    return merged


def score_regimes(
    factors: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> RegimeResult:
    """Compute seven transparent, exactly decomposable logistic regime scores.

    Missing factors are omitted and, by default, remaining contributions are
    rescaled by available absolute weight.  A score is suppressed when weighted
    coverage is below ``min_coverage``.  ``score_point_contribution`` values plus
    the 50-point neutral baseline exactly reproduce the reported 0--100 score.
    """

    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    cfg = _regime_config(config)
    data = _canonical_factor_frame(factors).sort_index()
    regimes = list(cfg["weights"])
    raw_output: dict[str, pd.Series] = {}
    score_output: dict[str, pd.Series] = {}
    coverage_output: dict[str, pd.Series] = {}
    contribution_parts: list[pd.DataFrame] = []
    temperature = float(cfg.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    for regime in regimes:
        weight_map = {
            factor: float(weight)
            for factor, weight in dict(cfg["weights"][regime]).items()
        }
        unknown = set(weight_map).difference(MVP_FACTOR_COLUMNS)
        if unknown:
            raise KeyError(f"unknown factors in weights for {regime}: {sorted(unknown)}")
        names = list(weight_map)
        weight = pd.Series(weight_map, dtype=float)
        values = data[names]
        availability = values.notna()
        total_weight = float(weight.abs().sum())
        if total_weight <= 0:
            raise ValueError(f"regime {regime!r} has no non-zero weights")
        available_weight = availability.mul(weight.abs(), axis=1).sum(axis=1)
        coverage = available_weight / total_weight
        scale = pd.Series(1.0, index=data.index)
        if bool(cfg.get("rescale_missing", True)):
            scale = total_weight / available_weight.replace(0.0, np.nan)
        weighted = values.mul(weight, axis=1).fillna(0.0).mul(scale, axis=0)
        intercept = float(dict(cfg.get("intercepts") or {}).get(regime, 0.0))
        raw_unclipped = weighted.sum(axis=1) + intercept
        valid = coverage >= float(cfg.get("min_coverage", 0.35))
        raw_unclipped = raw_unclipped.where(valid)
        raw_clip = cfg.get("raw_clip")
        raw = (
            raw_unclipped.clip(-abs(float(raw_clip)), abs(float(raw_clip)))
            if raw_clip is not None
            else raw_unclipped
        )
        score = pd.Series(
            100.0 * _sigmoid((raw / temperature).to_numpy()),
            index=data.index,
            dtype=float,
        ).where(raw.notna())
        raw_output[regime] = raw
        score_output[regime] = score
        coverage_output[regime] = coverage

        # If raw clipping activates, rescale linear contributions so the raw
        # contribution table continues to sum exactly to the displayed raw.
        components = weighted.copy()
        components["__intercept__"] = intercept
        component_sum = components.sum(axis=1)
        clip_scale = raw / component_sum.replace(0.0, np.nan)
        components = components.mul(clip_scale.fillna(1.0), axis=0)
        delta = score - 50.0
        point_multiplier = delta / raw.replace(0.0, np.nan)
        score_points = components.mul(point_multiplier.fillna(0.0), axis=0)
        for factor_name in components.columns:
            factor_value = (
                data[factor_name]
                if factor_name != "__intercept__"
                else pd.Series(1.0, index=data.index)
            )
            factor_weight = weight_map.get(factor_name, intercept)
            part = pd.DataFrame(
                {
                    "date": data.index,
                    "regime": regime,
                    "factor": factor_name,
                    "factor_value": factor_value.to_numpy(),
                    "weight": factor_weight,
                    "raw_contribution": components[factor_name].to_numpy(),
                    "score_point_contribution": score_points[factor_name].to_numpy(),
                    "available": factor_value.notna().to_numpy(),
                    "coverage": coverage.to_numpy(),
                }
            )
            part["absolute_raw_contribution"] = part["raw_contribution"].abs()
            contribution_parts.append(part)

    scores = pd.DataFrame(score_output, index=data.index)
    raw_scores = pd.DataFrame(raw_output, index=data.index)
    coverage_frame = pd.DataFrame(coverage_output, index=data.index)
    for output in (scores, raw_scores, coverage_frame):
        output.index.name = factors.index.name or "date"
    contributions = (
        pd.concat(contribution_parts, ignore_index=True)
        if contribution_parts
        else pd.DataFrame()
    )
    metadata = {
        "regime_labels": {
            name: dict(cfg.get("labels") or {}).get(
                name, REGIME_LABELS.get(name, name)
            )
            for name in regimes
        },
        "weights": cfg["weights"],
        "intercepts": cfg.get("intercepts", {}),
        "ignored_weights": cfg.get("ignored_weights", {}),
        "temperature": temperature,
        "min_coverage": cfg.get("min_coverage"),
        "point_in_time_safe": True,
    }
    return RegimeResult(scores, raw_scores, contributions, coverage_frame, metadata)


def compute_regime_scores(
    factors: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper returning only the 0--100 score DataFrame."""

    return score_regimes(factors, config).scores


def contribution_table(
    factors: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for explainability/reporting code."""

    result = score_regimes(factors, config)
    return result.latest_contributions if latest_only else result.contributions


def _softmax(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    safe = safe - np.max(safe, axis=1, keepdims=True)
    exp = np.exp(np.clip(safe, -700.0, 700.0))
    exp = np.where(finite, exp, 0.0)
    denominator = exp.sum(axis=1, keepdims=True)
    uniform = np.full_like(exp, 1.0 / exp.shape[1])
    return np.divide(exp, denominator, out=uniform, where=denominator > 0)


def _fit_softmax(
    x: np.ndarray,
    y: np.ndarray,
    classes: Sequence[Any],
    *,
    l2: float = 1.0,
    max_iter: int = 500,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free L2 multinomial-logit solver."""

    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y)
    class_list = list(classes)
    class_to_index = {label: index for index, label in enumerate(class_list)}
    encoded = np.array([class_to_index[item] for item in target], dtype=int)
    one_hot = np.eye(len(class_list), dtype=float)[encoded]
    rng = np.random.default_rng(int(seed))
    weights = rng.normal(0.0, 1.0e-5, size=(matrix.shape[1], len(class_list)))
    intercept = np.zeros(len(class_list), dtype=float)
    n = max(1, matrix.shape[0])
    for iteration in range(max(1, int(max_iter))):
        probabilities = _softmax(matrix @ weights + intercept)
        error = probabilities - one_hot
        gradient_w = matrix.T @ error / n + float(l2) * weights / n
        gradient_b = error.mean(axis=0)
        step = float(learning_rate) / np.sqrt(1.0 + iteration / 50.0)
        weights -= step * gradient_w
        intercept -= step * gradient_b
        if max(np.max(np.abs(gradient_w)), np.max(np.abs(gradient_b))) < 1.0e-7:
            break
    return weights, intercept


def _predict_softmax(x: np.ndarray, weights: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return _softmax(np.asarray(x, dtype=float) @ weights + intercept)


def _transparent_probabilities(
    factors: pd.DataFrame,
    transparent: RegimeResult | pd.DataFrame | None,
    *,
    temperature: float = 1.0,
) -> pd.DataFrame:
    if isinstance(transparent, RegimeResult):
        values = transparent.raw_scores
    elif isinstance(transparent, pd.DataFrame):
        values = transparent
    else:
        values = score_regimes(factors).raw_scores
    temperature = max(float(temperature), 1.0e-9)
    probabilities = _softmax(values.to_numpy(dtype=float) / temperature)
    return pd.DataFrame(probabilities, index=values.index, columns=values.columns)


DEFAULT_STATISTICAL_CONFIG: dict[str, Any] = {
    "min_train_size": 126,
    "min_class_observations": 5,
    "refit_every": 63,
    "l2": 1.0,
    "max_iter": 100,
    "learning_rate": 0.1,
    "random_seed": 42,
    "transparent_weight": 0.20,
    "transparent_temperature": 1.0,
}


def estimate_statistical_regimes(
    factors: pd.DataFrame,
    labels: pd.Series | pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    transparent: RegimeResult | pd.DataFrame | None = None,
) -> StatisticalRegimeResult:
    """Estimate causal walk-forward multinomial regime probabilities.

    Each prediction at date ``t`` is fit only on labelled observations before
    ``t``.  When labels or class histories are insufficient, probabilities
    fall back deterministically to the transparent model.  No optional machine
    learning dependency is required.
    """

    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    cfg = dict(DEFAULT_STATISTICAL_CONFIG)
    if config:
        cfg.update(dict(config))
    x_frame = _canonical_factor_frame(factors).sort_index()
    transparent_probs = _transparent_probabilities(
        x_frame,
        transparent,
        temperature=float(cfg.get("transparent_temperature", 1.0)),
    ).reindex(x_frame.index)

    if labels is None:
        return StatisticalRegimeResult(
            transparent_probs,
            pd.DataFrame(),
            pd.Series(dtype=float),
            "transparent_softmax_fallback",
            "labels were not supplied",
            {
                "random_seed": int(cfg["random_seed"]),
                "point_in_time_safe": True,
                "fallback_rows": len(x_frame),
            },
        )
    if isinstance(labels, pd.DataFrame):
        label_series = _safe_idxmax(labels)
    else:
        label_series = pd.Series(labels).copy()
    label_series = label_series.reindex(x_frame.index)
    observed_classes = list(pd.unique(label_series.dropna()))
    configured_classes = cfg.get("classes")
    if configured_classes is not None:
        classes = list(configured_classes)
    elif set(observed_classes).issubset(set(REGIME_NAMES)):
        classes = list(REGIME_NAMES)
    else:
        classes = sorted(observed_classes, key=str)
    if len(classes) < 2:
        fallback = transparent_probs.copy()
        return StatisticalRegimeResult(
            fallback,
            pd.DataFrame(),
            pd.Series(dtype=float),
            "transparent_softmax_fallback",
            "fewer than two labelled classes",
            {
                "random_seed": int(cfg["random_seed"]),
                "point_in_time_safe": True,
                "fallback_rows": len(x_frame),
            },
        )

    # Align fallback columns to the requested statistical classes.  Unknown
    # custom classes receive a small uniform prior.
    fallback = transparent_probs.reindex(columns=classes)
    fallback = fallback.fillna(1.0 / len(classes))
    fallback = fallback.div(fallback.sum(axis=1), axis=0)
    probabilities = fallback.copy()
    min_train = int(cfg.get("min_train_size", 126))
    min_class = int(cfg.get("min_class_observations", 5))
    refit_every = max(1, int(cfg.get("refit_every", 21)))
    alpha = float(np.clip(cfg.get("transparent_weight", 0.20), 0.0, 1.0))
    model: tuple[
        np.ndarray,
        np.ndarray,
        pd.Series,
        pd.Series,
        list[str],
        list[Any],
    ] | None = None
    latest_coefficients = pd.DataFrame()
    latest_intercepts = pd.Series(dtype=float)
    fallback_rows = 0
    fit_rows = 0
    last_fit_position = -refit_every

    for position, date in enumerate(x_frame.index):
        train_x = x_frame.iloc[:position]
        train_y = label_series.iloc[:position]
        valid_y = train_y.notna() & train_y.isin(classes)
        class_counts = train_y.loc[valid_y].value_counts()
        eligible_classes = [
            label for label in classes if int(class_counts.get(label, 0)) >= min_class
        ]
        enough = valid_y.sum() >= min_train and len(eligible_classes) >= 2
        # Refit only on the deterministic walk-forward cadence.  Treating the
        # current sample endpoint as a special refit date makes the prediction
        # at t depend on whether rows after t happen to be present in the
        # caller's frame, violating prefix invariance.
        should_refit = enough and (
            model is None or position - last_fit_position >= refit_every
        )
        if should_refit:
            use = valid_y & train_y.isin(eligible_classes)
            candidate = train_x.loc[use]
            feature_names = [
                name for name in candidate.columns if candidate[name].notna().sum() >= min_train // 2
            ]
            if feature_names:
                mean = candidate[feature_names].mean()
                scale = candidate[feature_names].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
                matrix = candidate[feature_names].fillna(mean).sub(mean).div(scale).to_numpy()
                fit_classes = [label for label in classes if label in eligible_classes]
                weights, intercept = _fit_softmax(
                    matrix,
                    train_y.loc[use].to_numpy(),
                    fit_classes,
                    l2=float(cfg.get("l2", 1.0)),
                    max_iter=int(cfg.get("max_iter", 500)),
                    learning_rate=float(cfg.get("learning_rate", 0.1)),
                    seed=int(cfg.get("random_seed", 42)),
                )
                model = (
                    weights,
                    intercept,
                    mean,
                    scale,
                    feature_names,
                    fit_classes,
                )
                latest_coefficients = pd.DataFrame(
                    weights, index=feature_names, columns=fit_classes
                )
                latest_intercepts = pd.Series(intercept, index=fit_classes, name="intercept")
                last_fit_position = position
                fit_rows += 1
            else:
                model = None

        if model is None:
            fallback_rows += 1
            continue
        weights, intercept, mean, scale, feature_names, model_classes = model
        row = x_frame.loc[[date], feature_names].fillna(mean).sub(mean).div(scale)
        statistical = pd.Series(
            _predict_softmax(row.to_numpy(), weights, intercept)[0],
            index=model_classes,
            dtype=float,
        ).reindex(classes, fill_value=0.0)
        blended = (1.0 - alpha) * statistical + alpha * fallback.loc[date]
        probabilities.loc[date] = blended / blended.sum()

    method = "walk_forward_multinomial_logit" if fit_rows else "transparent_softmax_fallback"
    reason = None if fit_rows else "insufficient labelled history for statistical fit"
    probabilities.index.name = factors.index.name or "date"
    probabilities.columns.name = "regime"
    return StatisticalRegimeResult(
        probabilities,
        latest_coefficients,
        latest_intercepts,
        method,
        reason,
        {
            "random_seed": int(cfg.get("random_seed", 42)),
            "point_in_time_safe": True,
            "fit_count": fit_rows,
            "fallback_rows": fallback_rows,
            "last_fit_position": (
                int(last_fit_position) if fit_rows else None
            ),
            "last_fit_date": (
                x_frame.index[last_fit_position] if fit_rows else None
            ),
            "classes": classes,
            "transparent_weight": alpha,
        },
    )


# Backwards-friendly aliases.
build_regime_scores = compute_regime_scores
transparent_regime_scores = score_regimes


__all__ = [
    "DEFAULT_REGIME_CONFIG",
    "DEFAULT_REGIME_WEIGHTS",
    "DEFAULT_STATISTICAL_CONFIG",
    "REGIME_LABELS",
    "REGIME_NAMES",
    "RegimeResult",
    "StatisticalRegimeResult",
    "build_regime_scores",
    "compute_regime_scores",
    "contribution_table",
    "estimate_statistical_regimes",
    "regime_config_from_mapping",
    "score_regimes",
    "transparent_regime_scores",
]
