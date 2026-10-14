"""Dependency-light lead/lag diagnostics for time-series research.

Lag direction is deliberately explicit throughout this module: a positive
lag ``k`` aligns ``x[t-k]`` with ``y[t]`` and therefore means that ``x`` leads
``y`` by ``k`` observations.  Negative lags are accepted only by the
descriptive cross-correlation scan and are marked as using future ``x``.
Regression and Granger designs reject negative lags.

These procedures measure temporal association and predictive precedence.
They do not, by themselves, identify structural or economic causality.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ASSOCIATION_DISCLAIMER = (
    "Lead/lag statistics measure association and predictive precedence, not "
    "structural causality; omitted variables, revisions, common drivers, and "
    "multiple testing can invalidate a causal interpretation."
)


@dataclass
class LeadLagAnalysisResult:
    """Tables produced by :func:`run_lead_lag_analysis`.

    The aliases exposed as properties make the result convenient in notebooks
    without changing the stable, descriptive field names.
    """

    cross_correlations: pd.DataFrame
    distributed_lag_coefficients: pd.DataFrame
    granger_tests: pd.DataFrame
    metadata: pd.DataFrame
    disclaimer: str = ASSOCIATION_DISCLAIMER

    @property
    def cross_correlation(self) -> pd.DataFrame:
        return self.cross_correlations

    @property
    def distributed_lag(self) -> pd.DataFrame:
        return self.distributed_lag_coefficients

    @property
    def granger(self) -> pd.DataFrame:
        return self.granger_tests

    def __getitem__(self, key: str) -> pd.DataFrame | str:
        aliases = {
            "cross_correlation": "cross_correlations",
            "correlations": "cross_correlations",
            "distributed_lag": "distributed_lag_coefficients",
            "coefficients": "distributed_lag_coefficients",
            "granger": "granger_tests",
        }
        return getattr(self, aliases.get(key, key))

    def to_dict(self) -> dict[str, pd.DataFrame | str]:
        return {
            "cross_correlations": self.cross_correlations,
            "distributed_lag_coefficients": self.distributed_lag_coefficients,
            "granger_tests": self.granger_tests,
            "metadata": self.metadata,
            "disclaimer": self.disclaimer,
        }


def _as_series(values: pd.Series | pd.DataFrame | Sequence[float], default_name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{default_name} DataFrame must have exactly one column")
        series = values.iloc[:, 0].copy()
    elif isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, name=default_name, dtype=float)
    if series.index.has_duplicates:
        series = series.loc[~series.index.duplicated(keep="last")]
    series = series.sort_index()
    series.name = str(series.name or default_name)
    return pd.to_numeric(series, errors="coerce")


def _joint_xy(
    y: pd.Series | pd.DataFrame | Sequence[float],
    x: pd.Series | pd.DataFrame | Sequence[float],
) -> tuple[pd.DataFrame, str, str]:
    y_series = _as_series(y, "y")
    x_series = _as_series(x, "x")
    y_name, x_name = str(y_series.name), str(x_series.name)
    frame = pd.concat([y_series.rename("__y__"), x_series.rename("__x__")], axis=1, join="inner")
    return frame.sort_index(), y_name, x_name


def _normalise_lags(
    lags: int | Iterable[int],
    *,
    signed_integer_range: bool = False,
    allow_negative: bool = True,
) -> list[int]:
    if isinstance(lags, (int, np.integer)):
        bound = int(lags)
        if bound < 0:
            raise ValueError("an integer lag bound must be non-negative")
        values = range(-bound, bound + 1) if signed_integer_range else range(0, bound + 1)
    else:
        values = lags
    try:
        normalised = sorted({int(value) for value in values})
    except (TypeError, ValueError) as exc:
        raise ValueError("lags must be an integer or an iterable of integers") from exc
    if not normalised:
        raise ValueError("at least one lag is required")
    if not allow_negative and normalised[0] < 0:
        raise ValueError("negative lags would use future x and are not allowed in a causal design")
    return normalised


def _student_t_pvalue(t_stat: float, degrees_freedom: int) -> float:
    if not np.isfinite(t_stat) or degrees_freedom <= 0:
        return np.nan
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(2.0 * student_t.sf(abs(t_stat), degrees_freedom))
    except ImportError:
        # Conservative-enough large-sample fallback when scipy is unavailable.
        return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for the incomplete beta."""

    max_iterations, epsilon, floor = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = floor if abs(d) < floor else d
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    return result


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    term = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - term * _beta_continued_fraction(b, a, 1.0 - x) / b


def _f_survival(f_stat: float, df_num: int, df_den: int) -> float:
    if not np.isfinite(f_stat) or f_stat < 0 or df_num <= 0 or df_den <= 0:
        return np.nan
    try:
        from scipy.stats import f as fisher_f  # type: ignore

        return float(fisher_f.sf(f_stat, df_num, df_den))
    except ImportError:
        argument = df_den / (df_den + df_num * f_stat)
        return float(_regularized_incomplete_beta(argument, df_den / 2.0, df_num / 2.0))


def cross_correlation_scan(
    y: pd.Series | pd.DataFrame | Sequence[float],
    x: pd.Series | pd.DataFrame | Sequence[float],
    lags: int | Iterable[int],
) -> pd.DataFrame:
    """Scan contemporaneous and lead/lag correlations on aligned observations.

    A positive lag ``k`` computes ``corr(y[t], x[t-k])``.  A negative lag is
    retained as a useful reverse-direction diagnostic, but its row is marked
    ``causal_design=False`` because it aligns current ``y`` with future ``x``.
    Passing an integer ``m`` scans every lag from ``-m`` through ``m``.
    """

    frame, y_name, x_name = _joint_xy(y, x)
    lag_values = _normalise_lags(lags, signed_integer_range=True)
    rows: list[dict[str, Any]] = []
    for lag in lag_values:
        sample = pd.concat(
            [frame["__y__"].rename("y"), frame["__x__"].shift(lag).rename("x_lagged")],
            axis=1,
        ).dropna()
        n_obs = int(len(sample))
        correlation = np.nan
        if n_obs >= 2 and float(sample["y"].std(ddof=0)) > 0 and float(sample["x_lagged"].std(ddof=0)) > 0:
            correlation = float(sample["y"].corr(sample["x_lagged"]))
        if n_obs > 2 and np.isfinite(correlation) and abs(correlation) < 1.0:
            t_stat = correlation * math.sqrt((n_obs - 2) / max(1.0e-15, 1.0 - correlation**2))
            standard_error = math.sqrt(max(0.0, 1.0 - correlation**2) / (n_obs - 2))
        elif n_obs > 2 and np.isfinite(correlation) and abs(correlation) == 1.0:
            t_stat = math.copysign(np.inf, correlation)
            standard_error = 0.0
        else:
            t_stat, standard_error = np.nan, np.nan
        direction = "x_leads_y" if lag > 0 else "contemporaneous" if lag == 0 else "y_leads_x"
        rows.append(
            {
                "dependent": y_name,
                "predictor": x_name,
                "lag": lag,
                "direction": direction,
                "n_obs": n_obs,
                "correlation": correlation,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "p_value": _student_t_pvalue(t_stat, n_obs - 2),
                "causal_design": lag >= 0,
                "uses_future_x": lag < 0,
                "association_not_causality": True,
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["lag_definition"] = "positive k means corr(y[t], x[t-k])"
    result.attrs["disclaimer"] = ASSOCIATION_DISCLAIMER
    return result


def _normalise_y_lags(y_lags: int | Iterable[int]) -> list[int]:
    if isinstance(y_lags, (int, np.integer)):
        if int(y_lags) < 0:
            raise ValueError("y_lags must be non-negative")
        return list(range(1, int(y_lags) + 1))
    values = sorted({int(value) for value in y_lags})
    if any(value <= 0 for value in values):
        raise ValueError("autoregressive y lags must be strictly positive")
    return values


def _controls_frame(controls: pd.Series | pd.DataFrame | None) -> pd.DataFrame:
    if controls is None:
        return pd.DataFrame()
    frame = controls.to_frame(controls.name or "control") if isinstance(controls, pd.Series) else controls.copy()
    if frame.columns.duplicated().any():
        raise ValueError("control column names must be unique")
    if frame.index.has_duplicates:
        frame = frame.loc[~frame.index.duplicated(keep="last")]
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def _build_distributed_lag_design(
    y: pd.Series | pd.DataFrame | Sequence[float],
    x: pd.Series | pd.DataFrame | Sequence[float],
    lags: list[int],
    *,
    y_lags: list[int],
    controls: pd.Series | pd.DataFrame | None,
    add_constant: bool,
) -> tuple[pd.Series, pd.DataFrame, dict[str, tuple[str, int | None]], str, str]:
    frame, y_name, x_name = _joint_xy(y, x)
    design = pd.DataFrame(index=frame.index)
    term_metadata: dict[str, tuple[str, int | None]] = {}
    if add_constant:
        design["constant"] = 1.0
        term_metadata["constant"] = ("intercept", None)
    for lag in lags:
        term = f"{x_name}_lag_{lag}"
        design[term] = frame["__x__"].shift(lag)
        term_metadata[term] = ("predictor", lag)
    for lag in y_lags:
        term = f"{y_name}_lag_{lag}"
        design[term] = frame["__y__"].shift(lag)
        term_metadata[term] = ("autoregressive", lag)
    control_frame = _controls_frame(controls)
    if not control_frame.empty:
        control_frame = control_frame.reindex(frame.index)
        for column in control_frame.columns:
            term = f"control_{column}"
            design[term] = control_frame[column]
            term_metadata[term] = ("control", 0)
    model = pd.concat([frame["__y__"].rename("response"), design], axis=1).dropna()
    return model["response"], model[design.columns], term_metadata, y_name, x_name


def _newey_west_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    *,
    rank: int,
    max_lag: int,
) -> np.ndarray:
    n_obs, n_terms = design.shape
    inverse = np.linalg.pinv(design.T @ design)
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(max_lag, n_obs - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    correction = n_obs / max(1, n_obs - rank)
    covariance = correction * inverse @ meat @ inverse
    return covariance.reshape((n_terms, n_terms))


def distributed_lag_regression(
    y: pd.Series | pd.DataFrame | Sequence[float],
    x: pd.Series | pd.DataFrame | Sequence[float],
    lags: int | Iterable[int],
    *,
    y_lags: int | Iterable[int] = 0,
    controls: pd.Series | pd.DataFrame | None = None,
    add_constant: bool = True,
    covariance: str = "newey_west",
    hac_lags: int | None = None,
    min_obs: int | None = None,
    include_design: bool = False,
) -> pd.DataFrame:
    """Fit ``y[t]`` on present/past ``x`` values with OLS inference.

    ``lags`` may be an iterable or a non-negative maximum lag (which expands
    to ``0..lags``).  Negative lags are rejected.  Optional autoregressive
    ``y_lags`` are strictly positive.  Controls are contemporaneous and must
    already be point-in-time available to the caller.  Newey-West HAC standard
    errors are the default; ``covariance='classical'`` requests homoskedastic
    OLS errors.  Set ``include_design=True`` to retain the aligned response and
    design matrix in ``result.attrs`` for audit/testing.
    """

    lag_values = _normalise_lags(lags, allow_negative=False)
    autoregressive_lags = _normalise_y_lags(y_lags)
    response, design, term_metadata, y_name, x_name = _build_distributed_lag_design(
        y,
        x,
        lag_values,
        y_lags=autoregressive_lags,
        controls=controls,
        add_constant=add_constant,
    )
    n_obs, n_terms = design.shape
    required = max(n_terms + 2, 8) if min_obs is None else max(n_terms + 1, int(min_obs))
    if n_obs < required:
        raise ValueError(f"insufficient complete observations: found {n_obs}, require at least {required}")
    design_values = design.to_numpy(dtype=float)
    response_values = response.to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design_values, response_values, rcond=None)
    fitted = design_values @ coefficients
    residuals = response_values - fitted
    ssr = float(residuals @ residuals)
    df_resid = n_obs - int(rank)
    covariance_name = covariance.lower().replace("-", "_")
    if covariance_name in {"newey_west", "hac"}:
        selected_hac_lags = (
            max(0, int(hac_lags))
            if hac_lags is not None
            else max(0, int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
        )
        covariance_matrix = _newey_west_covariance(
            design_values, residuals, rank=int(rank), max_lag=selected_hac_lags
        )
        covariance_name = "newey_west"
    elif covariance_name in {"classical", "ols", "homoskedastic"}:
        selected_hac_lags = 0
        sigma_squared = ssr / df_resid if df_resid > 0 else np.nan
        covariance_matrix = sigma_squared * np.linalg.pinv(design_values.T @ design_values)
        covariance_name = "classical"
    else:
        raise ValueError("covariance must be 'newey_west'/'hac' or 'classical'/'ols'")
    variances = np.diag(covariance_matrix)
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    centered_tss = float(np.square(response_values - response_values.mean()).sum())
    r_squared = 1.0 - ssr / centered_tss if centered_tss > 0 else np.nan
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (n_obs - 1) / df_resid
        if np.isfinite(r_squared) and df_resid > 0
        else np.nan
    )
    rows: list[dict[str, Any]] = []
    for position, term in enumerate(design.columns):
        term_type, lag = term_metadata[term]
        t_stat = float(t_statistics[position])
        std_error = float(standard_errors[position])
        rows.append(
            {
                "dependent": y_name,
                "predictor": x_name,
                "term": term,
                "term_type": term_type,
                "lag": lag,
                "coefficient": float(coefficients[position]),
                "standard_error": std_error,
                "t_stat": t_stat,
                "p_value": _student_t_pvalue(t_stat, df_resid),
                "n_obs": n_obs,
                "df_resid": df_resid,
                "rank": int(rank),
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "covariance": covariance_name,
                "hac_lags": selected_hac_lags,
                "causal_design": True,
                "uses_future_x": False,
                "association_not_causality": True,
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["lag_definition"] = "positive k means x[t-k] predicts y[t]"
    result.attrs["disclaimer"] = ASSOCIATION_DISCLAIMER
    result.attrs["fit_start"] = response.index.min()
    result.attrs["fit_end"] = response.index.max()
    if include_design:
        result.attrs["response"] = response.copy()
        result.attrs["design_matrix"] = design.copy()
        result.attrs["term_metadata"] = dict(term_metadata)
    return result


def _least_squares_ssr(response: np.ndarray, design: np.ndarray) -> tuple[float, int]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    return float(residuals @ residuals), int(rank)


def granger_causality_test(
    y: pd.Series | pd.DataFrame | Sequence[float],
    x: pd.Series | pd.DataFrame | Sequence[float],
    max_lag: int,
    *,
    add_constant: bool = True,
    alpha: float = 0.05,
    min_obs: int | None = None,
    include_design: bool = False,
) -> pd.DataFrame:
    """Test whether past ``x`` adds predictive power for ``y`` at each order.

    For order ``p``, the restricted model contains lags ``1..p`` of ``y`` and
    the unrestricted model adds lags ``1..p`` of ``x``.  Both models use the
    exact same rows.  The returned F-test is a predictive-precedence test and
    must not be interpreted as structural causality.
    """

    order_bound = int(max_lag)
    if order_bound < 1:
        raise ValueError("max_lag must be at least one")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    frame, y_name, x_name = _joint_xy(y, x)
    rows: list[dict[str, Any]] = []
    stored_designs: dict[int, dict[str, pd.Series | pd.DataFrame]] = {}
    for order in range(1, order_bound + 1):
        restricted = pd.DataFrame(index=frame.index)
        if add_constant:
            restricted["constant"] = 1.0
        for lag in range(1, order + 1):
            restricted[f"{y_name}_lag_{lag}"] = frame["__y__"].shift(lag)
        unrestricted = restricted.copy()
        for lag in range(1, order + 1):
            unrestricted[f"{x_name}_lag_{lag}"] = frame["__x__"].shift(lag)
        model = pd.concat([frame["__y__"].rename("response"), unrestricted], axis=1).dropna()
        response = model["response"]
        unrestricted_design = model[unrestricted.columns]
        restricted_design = unrestricted_design[restricted.columns]
        n_obs = int(len(model))
        n_unrestricted_terms = unrestricted_design.shape[1]
        required = (
            max(n_unrestricted_terms + 2, 8)
            if min_obs is None
            else max(n_unrestricted_terms + 1, int(min_obs))
        )
        base_row: dict[str, Any] = {
            "dependent": y_name,
            "predictor": x_name,
            "lag_order": order,
            "n_obs": n_obs,
            "df_num": order,
            "minimum_predictor_lag": 1,
            "causal_design": True,
            "uses_future_x": False,
            "association_not_causality": True,
        }
        if n_obs < required:
            rows.append(
                {
                    **base_row,
                    "df_den": n_obs - n_unrestricted_terms,
                    "ssr_restricted": np.nan,
                    "ssr_unrestricted": np.nan,
                    "incremental_r_squared": np.nan,
                    "f_stat": np.nan,
                    "p_value": np.nan,
                    "reject_null": False,
                    "status": "insufficient_observations",
                }
            )
            continue
        response_values = response.to_numpy(dtype=float)
        restricted_values = restricted_design.to_numpy(dtype=float)
        unrestricted_values = unrestricted_design.to_numpy(dtype=float)
        ssr_restricted, _ = _least_squares_ssr(response_values, restricted_values)
        ssr_unrestricted, rank_unrestricted = _least_squares_ssr(response_values, unrestricted_values)
        df_num = order
        df_den = n_obs - rank_unrestricted
        improvement = max(0.0, ssr_restricted - ssr_unrestricted)
        denominator = ssr_unrestricted / df_den if df_den > 0 else np.nan
        if np.isfinite(denominator) and denominator > 0:
            f_stat = (improvement / df_num) / denominator
        elif improvement > 0 and denominator == 0:
            f_stat = np.inf
        else:
            f_stat = np.nan
        p_value = _f_survival(f_stat, df_num, df_den)
        incremental = improvement / ssr_restricted if ssr_restricted > 0 else np.nan
        rows.append(
            {
                **base_row,
                "df_den": df_den,
                "ssr_restricted": ssr_restricted,
                "ssr_unrestricted": ssr_unrestricted,
                "incremental_r_squared": incremental,
                "f_stat": f_stat,
                "p_value": p_value,
                "reject_null": bool(np.isfinite(p_value) and p_value < alpha),
                "status": "ok",
            }
        )
        if include_design:
            stored_designs[order] = {
                "response": response.copy(),
                "restricted": restricted_design.copy(),
                "unrestricted": unrestricted_design.copy(),
            }
    result = pd.DataFrame(rows)
    result.attrs["null_hypothesis"] = "lags of x do not add predictive power for y"
    result.attrs["lag_definition"] = "only x[t-k], k>=1, enters the unrestricted model"
    result.attrs["disclaimer"] = ASSOCIATION_DISCLAIMER
    if include_design:
        result.attrs["designs"] = stored_designs
    return result


_DEFAULT_RELATIONSHIPS: tuple[tuple[str, str], ...] = (
    ("crop_condition", "weather_stress"),
    ("yield_revision", "crop_condition"),
    ("wheat", "weather_stress"),
    ("corn", "crop_condition"),
    ("urea", "natural_gas"),
    ("cf", "urea"),
    ("breakeven_5y", "wheat"),
)


def _parse_relationships(
    frame: pd.DataFrame,
    relationships: Sequence[tuple[str, str] | Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    if relationships is None:
        selected: list[tuple[str, str] | Mapping[str, Any]] = [
            pair for pair in _DEFAULT_RELATIONSHIPS if pair[0] in frame.columns and pair[1] in frame.columns
        ]
        if not selected and len(frame.columns) >= 2:
            selected = [(str(frame.columns[1]), str(frame.columns[0]))]
    elif isinstance(relationships, Mapping):
        if any(key in relationships for key in ("y", "dependent", "x", "predictor")):
            selected = [relationships]
        else:
            selected = []
            for label, specification in relationships.items():
                if isinstance(specification, Mapping):
                    selected.append({**specification, "label": str(label)})
                else:
                    dependent, predictor = specification
                    selected.append({"dependent": dependent, "predictor": predictor, "label": str(label)})
    else:
        selected = list(relationships)
    parsed: list[dict[str, str]] = []
    for position, relationship in enumerate(selected):
        if isinstance(relationship, Mapping):
            dependent = relationship.get("dependent", relationship.get("y"))
            predictor = relationship.get("predictor", relationship.get("x"))
            label = relationship.get("label")
        else:
            if len(relationship) != 2:
                raise ValueError("relationship tuples must be (dependent/y, predictor/x)")
            dependent, predictor = relationship
            label = None
        if dependent is None or predictor is None:
            raise ValueError("every relationship requires dependent/y and predictor/x")
        dependent, predictor = str(dependent), str(predictor)
        if dependent not in frame.columns or predictor not in frame.columns:
            raise KeyError(f"relationship columns not found: dependent={dependent!r}, predictor={predictor!r}")
        parsed.append(
            {
                "dependent": dependent,
                "predictor": predictor,
                "relationship": str(label or f"{predictor} -> {dependent}"),
                "relationship_order": str(position),
            }
        )
    if not parsed:
        raise ValueError("no analyzable relationships were supplied or discovered")
    return parsed


def _transform_pair(series: pd.Series, transform: str) -> pd.Series:
    name = series.name
    transform_name = transform.lower().replace("-", "_")
    numeric = pd.to_numeric(series, errors="coerce")
    if transform_name in {"level", "levels", "none"}:
        result = numeric
    elif transform_name in {"difference", "differences", "diff"}:
        result = numeric.diff()
    elif transform_name in {"return", "returns", "pct_change"}:
        result = numeric.pct_change(fill_method=None)
    elif transform_name in {"log_return", "log_returns", "log_difference"}:
        result = np.log(numeric.where(numeric > 0)).diff()
    else:
        raise ValueError("transform must be levels, differences, returns, or log_returns")
    result.name = name
    return result


def run_lead_lag_analysis(
    frame: pd.DataFrame,
    relationships: Sequence[tuple[str, str] | Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    lags: int | Iterable[int] | None = None,
    max_lag: int = 5,
    transform: str = "differences",
    covariance: str = "newey_west",
    hac_lags: int | None = None,
    alpha: float = 0.05,
    min_obs: int | None = None,
) -> LeadLagAnalysisResult:
    """Run correlation, distributed-lag, and Granger tables per relationship.

    Relationship tuples are ordered ``(dependent/y, predictor/x)``.  Mapping
    specifications may instead use ``dependent``/``predictor`` or ``y``/``x``
    and an optional label.  When omitted, a compact set of economically
    motivated agricultural relationships present in ``frame`` is used.

    The default first-difference transform reduces, but cannot eliminate,
    spurious inference from trending levels.  All outputs are deterministic;
    callers remain responsible for point-in-time source alignment and for
    correcting significance thresholds across a broader research program.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    working = frame.copy()
    if "date" in working.columns and not isinstance(working.index, pd.DatetimeIndex):
        working = working.set_index("date")
    if working.index.has_duplicates:
        working = working.loc[~working.index.duplicated(keep="last")]
    working = working.sort_index()
    relationship_specs = _parse_relationships(working, relationships)
    lag_bound = int(max_lag)
    if lag_bound < 1:
        raise ValueError("max_lag must be at least one")
    scan_lags = (
        list(range(-lag_bound, lag_bound + 1))
        if lags is None
        else _normalise_lags(lags, signed_integer_range=True)
    )
    regression_lags = sorted({lag for lag in scan_lags if lag >= 0})
    if not regression_lags:
        raise ValueError("lags must contain at least one non-negative value for distributed-lag regression")

    correlation_tables: list[pd.DataFrame] = []
    coefficient_tables: list[pd.DataFrame] = []
    granger_tables: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for relationship in relationship_specs:
        dependent = relationship["dependent"]
        predictor = relationship["predictor"]
        label = relationship["relationship"]
        y = _transform_pair(working[dependent].rename(dependent), transform)
        x = _transform_pair(working[predictor].rename(predictor), transform)
        complete_input = pd.concat([y, x], axis=1).dropna()

        correlations = cross_correlation_scan(y, x, scan_lags).copy()
        coefficients = distributed_lag_regression(
            y,
            x,
            regression_lags,
            covariance=covariance,
            hac_lags=hac_lags,
            min_obs=min_obs,
        ).copy()
        granger = granger_causality_test(
            y,
            x,
            lag_bound,
            alpha=alpha,
            min_obs=min_obs,
        ).copy()
        for table in (correlations, coefficients, granger):
            table.attrs = {}
            table.insert(0, "relationship", label)
        correlation_tables.append(correlations)
        coefficient_tables.append(coefficients)
        granger_tables.append(granger)
        metadata_rows.append(
            {
                "relationship": label,
                "dependent": dependent,
                "predictor": predictor,
                "transform": transform,
                "input_observations": int(len(working)),
                "complete_pair_observations": int(len(complete_input)),
                "scan_min_lag": min(scan_lags),
                "scan_max_lag": max(scan_lags),
                "distributed_lags": tuple(regression_lags),
                "granger_max_lag": lag_bound,
                "alpha": float(alpha),
                "point_in_time_alignment_required": True,
                "association_not_causality": True,
            }
        )
    result = LeadLagAnalysisResult(
        cross_correlations=pd.concat(correlation_tables, ignore_index=True),
        distributed_lag_coefficients=pd.concat(coefficient_tables, ignore_index=True),
        granger_tests=pd.concat(granger_tables, ignore_index=True),
        metadata=pd.DataFrame(metadata_rows),
    )
    for table in (
        result.cross_correlations,
        result.distributed_lag_coefficients,
        result.granger_tests,
        result.metadata,
    ):
        table.attrs["disclaimer"] = ASSOCIATION_DISCLAIMER
    return result


__all__ = [
    "ASSOCIATION_DISCLAIMER",
    "LeadLagAnalysisResult",
    "cross_correlation_scan",
    "distributed_lag_regression",
    "granger_causality_test",
    "run_lead_lag_analysis",
]
