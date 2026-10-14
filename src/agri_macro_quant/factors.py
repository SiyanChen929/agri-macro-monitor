"""Construction of the ten minimum-viable agricultural macro factors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .features import robust_zscore, rolling_zscore, rolling_winsorize


MVP_FACTOR_COLUMNS: tuple[str, ...] = (
    "crop_condition",
    "yield_revision",
    "exporter_stocks_to_use",
    "policy_stress",
    "weather_stress",
    "corn_urea",
    "dxy",
    "us2y",
    "breakeven",
    "wheat_momentum",
)

MVP_FACTOR_LABELS: dict[str, str] = {
    "crop_condition": "F1 Crop Condition",
    "yield_revision": "F2 Yield Revision",
    "exporter_stocks_to_use": "F3 Exporter Stocks-to-Use",
    "policy_stress": "F4 Policy Stress",
    "weather_stress": "F5 Weather Stress",
    "corn_urea": "F6 Corn/Urea",
    "dxy": "F7 DXY",
    "us2y": "F8 US 2Y",
    "breakeven": "F9 Breakeven",
    "wheat_momentum": "F10 Wheat Momentum",
}


DEFAULT_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "crop_condition": (
        "crop_condition",
        "crop_conditions",
        "crop_condition_good_excellent",
        "crop_condition_rating",
        "condition_rating",
    ),
    "yield_revision": (
        "yield_revision",
        "yield_revisions",
        "wasde_yield_revision",
        "crop_yield_revision",
    ),
    "exporter_stocks_to_use": (
        "exporter_stocks_to_use",
        "exporter_stock_to_use",
        "exporter_stu",
        "exporter_stocks_use",
        "exportable_surplus",
    ),
    "policy_stress": (
        "policy_stress",
        "policy_stress_index",
        "trade_policy_stress",
        "export_restriction_index",
    ),
    "weather_stress": (
        "weather_stress",
        "weather_stress_index",
        "drought_heat_stress",
        "drought_stress",
    ),
    "corn_urea": (
        "corn_urea",
        "corn_urea_ratio",
        "corn_to_urea",
        "fertilizer_affordability",
    ),
    "corn": ("corn", "corn_price", "corn_futures", "zc", "zc1"),
    "urea": ("urea", "urea_price", "urea_fob", "nitrogen_urea"),
    "dxy": ("dxy", "dollar_index", "usd_index", "dtwexbgs"),
    "us2y": (
        "us2y",
        "us_2y",
        "us2y_yield",
        "treasury_2y",
        "two_year_yield",
        "dgs2",
    ),
    "breakeven": (
        "breakeven",
        "breakeven_5y",
        "breakeven_inflation",
        "five_year_breakeven",
        "us5y_breakeven",
        "t5yie",
    ),
    "wheat_momentum": (
        "wheat_momentum",
        "wheat_mom",
        "wheat_return_60",
        "wheat_60d_momentum",
    ),
    "wheat": ("wheat", "wheat_price", "wheat_futures", "zw", "zw1"),
}


DEFAULT_FACTOR_CONFIG: dict[str, Any] = {
    "sources": {},
    "aliases": {},
    "normalization": "robust_zscore",
    "normalization_window": 252,
    "min_periods": 20,
    "history_lag": 0,
    "winsor_limits": (0.01, 0.99),
    "z_clip": 4.0,
    "wheat_momentum_window": 60,
    "corn_urea_log_ratio": False,
    "include_raw": False,
    "strict": False,
}


def _token(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _resolve_column(
    frame: pd.DataFrame,
    key: str,
    sources: Mapping[str, Any],
    aliases: Mapping[str, Sequence[str]],
) -> Any | None:
    explicit = sources.get(key)
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(f"configured source {explicit!r} for {key!r} is not in frame")
        return explicit
    candidates = list(aliases.get(key, ()))
    candidates.extend(DEFAULT_SOURCE_ALIASES.get(key, (key,)))
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    token_map: dict[str, Any] = {}
    for column in frame.columns:
        token_map.setdefault(_token(column), column)
    for candidate in candidates:
        if _token(candidate) in token_map:
            return token_map[_token(candidate)]
    return None


def _numeric(frame: pd.DataFrame, column: Any | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_FACTOR_CONFIG)
    if config:
        result.update(dict(config))
    return result


def build_raw_mvp_factors(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Map a wide input frame into the ten unnormalised MVP factor series.

    Direct input columns win over derived versions.  If ``corn_urea`` is not
    supplied it is computed from corn and urea prices.  If ``wheat_momentum``
    is not supplied it is computed as a trailing log return (or arithmetic
    change for a non-positive series).
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    cfg = _config(config)
    data = frame.copy().sort_index()
    sources = dict(cfg.get("sources") or {})
    aliases = dict(cfg.get("aliases") or {})
    resolved: dict[str, Any | None] = {}
    for key in set(MVP_FACTOR_COLUMNS).union({"corn", "urea", "wheat"}):
        resolved[key] = _resolve_column(data, key, sources, aliases)

    raw: dict[str, pd.Series] = {}
    for name in MVP_FACTOR_COLUMNS:
        if name not in {"corn_urea", "wheat_momentum"}:
            raw[name] = _numeric(data, resolved[name])

    if resolved["corn_urea"] is not None:
        raw["corn_urea"] = _numeric(data, resolved["corn_urea"])
    else:
        corn = _numeric(data, resolved["corn"])
        urea = _numeric(data, resolved["urea"])
        ratio = corn / urea.mask(urea.abs() <= 1.0e-12)
        if bool(cfg.get("corn_urea_log_ratio", False)):
            ratio = np.log(ratio.where(ratio > 0))
        raw["corn_urea"] = ratio.replace([np.inf, -np.inf], np.nan)

    if resolved["wheat_momentum"] is not None:
        raw["wheat_momentum"] = _numeric(data, resolved["wheat_momentum"])
    else:
        wheat = _numeric(data, resolved["wheat"])
        window = int(cfg.get("wheat_momentum_window", 60))
        previous = wheat.shift(window)
        log_momentum = np.log(
            wheat.where(wheat > 0) / previous.where(previous > 0)
        )
        momentum = log_momentum.where(
            (wheat > 0) & (previous > 0), wheat - previous
        )
        raw["wheat_momentum"] = momentum.replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame(raw, index=data.index).reindex(columns=MVP_FACTOR_COLUMNS)
    missing_sources = [
        name for name in MVP_FACTOR_COLUMNS if result[name].notna().sum() == 0
    ]
    if missing_sources and bool(cfg.get("strict", False)):
        raise KeyError(f"no usable source data for factors: {missing_sources}")
    result.index.name = frame.index.name or "date"
    result.attrs["source_columns"] = {
        name: resolved.get(name) for name in MVP_FACTOR_COLUMNS
    }
    result.attrs["missing_factors"] = missing_sources
    return result


def _factor_setting(cfg: Mapping[str, Any], name: str, key: str, default: Any) -> Any:
    factor_overrides = cfg.get("factors") or {}
    definition = factor_overrides.get(name, {}) if isinstance(factor_overrides, Mapping) else {}
    return definition.get(key, default) if isinstance(definition, Mapping) else default


def build_mvp_factors(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return the standardized ten-factor MVP frame.

    Factor direction is intentionally economic rather than uniformly
    "stress-positive": high crop condition and high stocks remain positive raw
    concepts; the regime weight table applies the appropriate negative signs.
    This preserves interpretability of factor values and contributions.
    """

    cfg = _config(config)
    raw = build_raw_mvp_factors(frame, cfg)
    output: dict[str, pd.Series] = {}
    for name in MVP_FACTOR_COLUMNS:
        x = raw[name]
        method = str(
            _factor_setting(cfg, name, "normalization", cfg.get("normalization"))
        ).lower()
        window = int(
            _factor_setting(
                cfg, name, "window", cfg.get("normalization_window", 252)
            )
        )
        min_periods = min(
            window,
            max(
                1,
                int(
                    _factor_setting(
                        cfg, name, "min_periods", cfg.get("min_periods", 20)
                    )
                ),
            ),
        )
        history_lag = int(
            _factor_setting(cfg, name, "history_lag", cfg.get("history_lag", 0))
        )
        clip = _factor_setting(cfg, name, "z_clip", cfg.get("z_clip", 4.0))
        if method in {"none", "identity", "raw"}:
            factor = x.copy()
        elif method in {"zscore", "rolling_zscore", "standard"}:
            factor = rolling_zscore(
                x,
                window,
                min_periods=min_periods,
                history_lag=history_lag,
                clip=clip,
            )
        elif method in {"winsor_zscore", "winsorized_zscore"}:
            winsorized = rolling_winsorize(
                x,
                window,
                cfg.get("winsor_limits", (0.01, 0.99)),
                min_periods=min_periods,
                history_lag=history_lag,
            )
            factor = rolling_zscore(
                winsorized,
                window,
                min_periods=min_periods,
                history_lag=history_lag,
                clip=clip,
            )
        elif method in {"robust", "robust_zscore", "rolling_robust_zscore"}:
            factor = robust_zscore(
                x,
                window,
                min_periods=min_periods,
                history_lag=history_lag,
                winsor_limits=cfg.get("winsor_limits"),
                clip=clip,
            )
        else:
            raise ValueError(f"unsupported normalization method {method!r} for {name}")
        direction = float(_factor_setting(cfg, name, "direction", 1.0))
        output[name] = direction * factor
        if bool(cfg.get("include_raw", False)):
            output[f"raw__{name}"] = raw[name]

    result = pd.DataFrame(output, index=raw.index)
    result.index.name = raw.index.name
    result.attrs["factor_config"] = cfg
    result.attrs["missing_factors"] = raw.attrs.get("missing_factors", [])
    result.attrs["source_columns"] = raw.attrs.get("source_columns", {})
    result.attrs["point_in_time_safe"] = True
    return result


def factor_coverage(factors: pd.DataFrame) -> pd.DataFrame:
    """Return per-date availability and overall observations for each factor."""

    available = factors.reindex(columns=MVP_FACTOR_COLUMNS).notna()
    summary = pd.DataFrame(
        {
            "observations": available.sum(axis=0).astype(int),
            "coverage_fraction": available.mean(axis=0),
            "latest_available": available.iloc[-1] if len(available) else False,
        }
    )
    summary.index.name = "factor"
    return summary


# Concise aliases for orchestration code.
construct_mvp_factors = build_mvp_factors
make_factors = build_mvp_factors


__all__ = [
    "DEFAULT_FACTOR_CONFIG",
    "DEFAULT_SOURCE_ALIASES",
    "MVP_FACTOR_COLUMNS",
    "MVP_FACTOR_LABELS",
    "build_mvp_factors",
    "build_raw_mvp_factors",
    "construct_mvp_factors",
    "factor_coverage",
    "make_factors",
]
