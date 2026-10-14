"""Causal regime-transition probability estimation with robust fallbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .regimes import (
    REGIME_NAMES,
    RegimeResult,
    _fit_softmax,
    _predict_softmax,
    _safe_idxmax,
)


DEFAULT_HORIZONS: dict[str, int] = {"1w": 5, "1m": 21, "3m": 63}

DEFAULT_TRANSITION_CONFIG: dict[str, Any] = {
    "horizons": DEFAULT_HORIZONS,
    "smoothing": 1.0,
    "min_train_size": 126,
    "min_class_observations": 5,
    "refit_every": 63,
    "l2": 1.0,
    "max_iter": 100,
    "learning_rate": 0.1,
    "random_seed": 42,
    "empirical_weight": 0.25,
}


@dataclass(frozen=True)
class TransitionResult:
    """Transition matrices plus causal conditional forecast histories."""

    transition_matrices: dict[str, pd.DataFrame]
    transition_counts: dict[str, pd.DataFrame]
    forecast_history: dict[str, pd.DataFrame]
    current_regime: pd.Series
    model_coefficients: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_probabilities(self) -> pd.DataFrame:
        """Latest current-to-next probability vector for each horizon."""

        rows: dict[str, pd.Series] = {}
        for horizon, history in self.forecast_history.items():
            if not history.empty:
                rows[horizon] = history.iloc[-1]
        result = pd.DataFrame(rows).T
        result.index.name = "horizon"
        result.columns.name = "next_regime"
        return result

    @property
    def latest_forecasts(self) -> pd.DataFrame:
        """Long-form latest probabilities convenient for reports."""

        wide = self.latest_probabilities
        if wide.empty:
            return pd.DataFrame(columns=["horizon", "next_regime", "probability"])
        return (
            wide.stack()
            .rename("probability")
            .reset_index()
            .sort_values(["horizon", "probability"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def matrix(self, horizon: str = "1m") -> pd.DataFrame:
        """Return the empirical current-regime x next-regime matrix."""

        key = _resolve_horizon_key(horizon, self.transition_matrices)
        return self.transition_matrices[key]

    def latest_for(self, horizon: str = "1m") -> pd.Series:
        """Return the latest conditional next-regime vector."""

        key = _resolve_horizon_key(horizon, self.forecast_history)
        result = self.forecast_history[key].iloc[-1].sort_values(ascending=False)
        result.name = key
        return result


def _resolve_horizon_key(value: str, available: Mapping[str, Any]) -> str:
    if value in available:
        return value
    token = "".join(character for character in str(value).lower() if character.isalnum())
    aliases = {
        "1week": "1w",
        "oneweek": "1w",
        "5d": "1w",
        "1month": "1m",
        "onemonth": "1m",
        "21d": "1m",
        "3month": "3m",
        "3months": "3m",
        "threemonths": "3m",
        "63d": "3m",
    }
    candidate = aliases.get(token)
    if candidate in available:
        return candidate
    raise KeyError(f"unknown horizon {value!r}; available: {list(available)}")


def _labels_from_regimes(
    regimes: RegimeResult | pd.DataFrame | pd.Series | Sequence[Any],
) -> tuple[pd.Series, list[Any]]:
    if isinstance(regimes, RegimeResult):
        scores = regimes.scores
        labels = _safe_idxmax(scores)
        states = list(scores.columns)
    elif isinstance(regimes, pd.DataFrame):
        numeric = regimes.apply(pd.to_numeric, errors="coerce")
        labels = _safe_idxmax(numeric)
        states = list(numeric.columns)
    elif isinstance(regimes, pd.Series):
        labels = regimes.copy()
        states = list(pd.unique(labels.dropna()))
    else:
        labels = pd.Series(regimes)
        states = list(pd.unique(labels.dropna()))
    labels = labels.sort_index()
    return labels, states


def _normalise_horizons(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        horizons = {str(name): int(periods) for name, periods in value.items()}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        horizons = {f"{int(periods)}d": int(periods) for periods in value}
    else:
        raise TypeError("horizons must be a mapping or sequence of positive integers")
    if not horizons or any(periods < 1 for periods in horizons.values()):
        raise ValueError("all transition horizons must be positive")
    return horizons


def _posterior_matrix(counts: pd.DataFrame, smoothing: float) -> pd.DataFrame:
    alpha = float(smoothing)
    if alpha < 0:
        raise ValueError("smoothing cannot be negative")
    numerator = counts.astype(float) + alpha
    denominator = numerator.sum(axis=1).replace(0.0, np.nan)
    result = numerator.div(denominator, axis=0)
    if result.isna().any(axis=None):
        result = result.fillna(1.0 / max(1, result.shape[1]))
    result.index.name = "current_regime"
    result.columns.name = "next_regime"
    return result


def _empirical_estimates(
    labels: pd.Series,
    states: list[Any],
    steps: int,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(0, index=states, columns=states, dtype=int)
    history = pd.DataFrame(np.nan, index=labels.index, columns=states, dtype=float)
    values = labels.to_numpy(dtype=object)
    for position, current in enumerate(values):
        completed_origin = position - steps
        if completed_origin >= 0:
            origin = values[completed_origin]
            destination = current
            if pd.notna(origin) and pd.notna(destination):
                if origin in counts.index and destination in counts.columns:
                    counts.loc[origin, destination] += 1
        posterior = _posterior_matrix(counts, smoothing)
        if pd.notna(current) and current in posterior.index:
            history.iloc[position] = posterior.loc[current]
        else:
            destination_counts = counts.sum(axis=0).astype(float) + smoothing
            total = destination_counts.sum()
            history.iloc[position] = (
                destination_counts / total
                if total > 0
                else np.full(len(states), 1.0 / len(states))
            )
    return counts, _posterior_matrix(counts, smoothing), history


def _numeric_features(
    features: pd.DataFrame | None,
    index: pd.Index,
    columns: Sequence[Any] | None,
) -> pd.DataFrame | None:
    if features is None:
        return None
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    data = features.reindex(index)
    if columns is not None:
        missing = set(columns).difference(data.columns)
        if missing:
            raise KeyError(f"transition feature columns missing: {sorted(missing, key=str)}")
        data = data[list(columns)]
    data = data.select_dtypes(include=[np.number, "bool"]).astype(float)
    return data.replace([np.inf, -np.inf], np.nan)


def _conditional_history(
    labels: pd.Series,
    features: pd.DataFrame,
    states: list[Any],
    steps: int,
    empirical: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history = empirical.copy()
    min_train = max(2, int(cfg.get("min_train_size", 126)))
    min_class = max(1, int(cfg.get("min_class_observations", 5)))
    refit_every = max(1, int(cfg.get("refit_every", 21)))
    blend_weight = float(np.clip(cfg.get("empirical_weight", 0.25), 0.0, 1.0))
    seed = int(cfg.get("random_seed", 42))
    model: dict[str, Any] | None = None
    last_fit = -refit_every
    fit_count = 0
    statistical_rows = 0
    latest_coefficients = pd.DataFrame()
    values = labels.to_numpy(dtype=object)

    for position, date in enumerate(labels.index):
        final_origin = position - steps
        enough_pairs = final_origin + 1 >= min_train
        # A refit is triggered solely by the deterministic walk-forward
        # cadence.  An endpoint-triggered refit would make a forecast at t
        # change merely because the caller later appends observations.
        if enough_pairs and (
            model is None or position - last_fit >= refit_every
        ):
            origins = np.arange(final_origin + 1)
            destinations = origins + steps
            origin_labels = pd.Series(values[origins], index=origins)
            target_labels = pd.Series(values[destinations], index=origins)
            valid = origin_labels.notna() & target_labels.notna()
            valid &= origin_labels.isin(states) & target_labels.isin(states)
            counts = target_labels.loc[valid].value_counts()
            eligible = [state for state in states if counts.get(state, 0) >= min_class]
            use = valid & target_labels.isin(eligible)
            if int(use.sum()) >= min_train and len(eligible) >= 2:
                use_positions = origins[use.to_numpy()]
                candidate = features.iloc[use_positions]
                feature_names = [
                    str(column)
                    for column in candidate.columns
                    if candidate[column].notna().sum() >= max(2, int(use.sum()) // 2)
                ]
                # Resolve names back to original column objects after the
                # reporting-friendly string conversion.
                name_lookup = {str(column): column for column in candidate.columns}
                feature_columns = [name_lookup[name] for name in feature_names]
                mean = candidate[feature_columns].mean()
                scale = (
                    candidate[feature_columns]
                    .std(ddof=0)
                    .replace(0.0, 1.0)
                    .fillna(1.0)
                )
                continuous = (
                    candidate[feature_columns]
                    .fillna(mean)
                    .sub(mean)
                    .div(scale)
                    .to_numpy(dtype=float)
                )
                state_design = np.column_stack(
                    [origin_labels.loc[use].to_numpy() == state for state in states]
                ).astype(float)
                design = np.column_stack([continuous, state_design])
                model_classes = [state for state in states if state in eligible]
                weights, intercept = _fit_softmax(
                    design,
                    target_labels.loc[use].to_numpy(),
                    model_classes,
                    l2=float(cfg.get("l2", 1.0)),
                    max_iter=int(cfg.get("max_iter", 500)),
                    learning_rate=float(cfg.get("learning_rate", 0.1)),
                    seed=seed + int(steps),
                )
                coefficient_names = feature_names + [f"current={state}" for state in states]
                latest_coefficients = pd.DataFrame(
                    weights, index=coefficient_names, columns=model_classes
                )
                model = {
                    "weights": weights,
                    "intercept": intercept,
                    "classes": model_classes,
                    "feature_columns": feature_columns,
                    "mean": mean,
                    "scale": scale,
                }
                last_fit = position
                fit_count += 1

        current = values[position]
        if model is None or pd.isna(current):
            continue
        row = (
            features.loc[[date], model["feature_columns"]]
            .fillna(model["mean"])
            .sub(model["mean"])
            .div(model["scale"])
            .to_numpy(dtype=float)
        )
        state_row = np.array([[float(current == state) for state in states]])
        design_row = np.column_stack([row, state_row])
        model_probability = pd.Series(
            _predict_softmax(
                design_row, model["weights"], model["intercept"]
            )[0],
            index=model["classes"],
            dtype=float,
        ).reindex(states, fill_value=0.0)
        fallback = empirical.loc[date]
        blended = (1.0 - blend_weight) * model_probability + blend_weight * fallback
        history.loc[date] = blended / blended.sum()
        statistical_rows += 1

    return latest_coefficients, history, {
        "fit_count": fit_count,
        "statistical_rows": statistical_rows,
        "fallback_rows": int(len(history) - statistical_rows),
        "last_fit_position": int(last_fit) if fit_count else None,
        "last_fit_date": labels.index[last_fit] if fit_count else None,
        "method": (
            "walk_forward_conditional_multinomial_logit"
            if fit_count
            else "dirichlet_empirical_fallback"
        ),
    }


def estimate_transition_probabilities(
    regimes: RegimeResult | pd.DataFrame | pd.Series | Sequence[Any],
    features: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> TransitionResult:
    """Estimate 1-week, 1-month and 3-month transition probabilities.

    Returned full-sample matrices are Dirichlet-smoothed empirical estimates as
    of the final date.  ``forecast_history`` is causal at every row.  If feature
    history is adequate, its forecasts use expanding walk-forward conditional
    multinomial logits; otherwise the empirical probabilities are retained.
    """

    cfg = dict(DEFAULT_TRANSITION_CONFIG)
    if config:
        cfg.update(dict(config))
    horizons = _normalise_horizons(cfg.get("horizons", DEFAULT_HORIZONS))
    labels, inferred_states = _labels_from_regimes(regimes)
    configured_states = cfg.get("states")
    if configured_states is not None:
        states = list(configured_states)
    elif set(inferred_states).issubset(set(REGIME_NAMES)):
        states = list(REGIME_NAMES)
    else:
        states = inferred_states
    if not states:
        raise ValueError("cannot estimate transitions without at least one regime state")
    unknown_labels = set(labels.dropna()).difference(states)
    if unknown_labels:
        raise ValueError(f"labels not present in configured states: {sorted(unknown_labels, key=str)}")
    feature_frame = _numeric_features(features, labels.index, cfg.get("feature_columns"))

    matrices: dict[str, pd.DataFrame] = {}
    counts_output: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    coefficients: dict[str, pd.DataFrame] = {}
    horizon_metadata: dict[str, Any] = {}
    for name, steps in horizons.items():
        counts, matrix, empirical_history = _empirical_estimates(
            labels, states, steps, float(cfg.get("smoothing", 1.0))
        )
        counts_output[name] = counts
        matrices[name] = matrix
        if feature_frame is not None and feature_frame.shape[1] > 0:
            coefficient, history, model_meta = _conditional_history(
                labels, feature_frame, states, steps, empirical_history, cfg
            )
            coefficients[name] = coefficient
            histories[name] = history
            horizon_metadata[name] = model_meta
        else:
            histories[name] = empirical_history
            coefficients[name] = pd.DataFrame()
            horizon_metadata[name] = {
                "fit_count": 0,
                "statistical_rows": 0,
                "fallback_rows": len(labels),
                "method": "dirichlet_empirical_fallback",
                "fallback_reason": "features were not supplied",
            }
        histories[name].index.name = labels.index.name or "date"
        histories[name].columns.name = "next_regime"

    labels.name = "current_regime"
    metadata = {
        "horizons": horizons,
        "states": states,
        "smoothing": float(cfg.get("smoothing", 1.0)),
        "random_seed": int(cfg.get("random_seed", 42)),
        "point_in_time_safe_forecasts": True,
        "models": horizon_metadata,
    }
    return TransitionResult(
        matrices,
        counts_output,
        histories,
        labels,
        coefficients,
        metadata,
    )


# Concise aliases for orchestration and earlier notebooks.
fit_transition_model = estimate_transition_probabilities
transition_probabilities = estimate_transition_probabilities


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_TRANSITION_CONFIG",
    "TransitionResult",
    "estimate_transition_probabilities",
    "fit_transition_model",
    "transition_probabilities",
]
