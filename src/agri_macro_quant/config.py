"""YAML-backed, typed configuration with project-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import os
import math

import pandas as pd
import yaml

from .data_models import SeriesSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _reject_unknown_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(map(str, values)).difference(allowed))
    if unknown:
        raise ValueError(
            f"{context} contains unknown configuration key(s): {unknown}. "
            "Unknown keys are rejected so safety-critical typos cannot silently use defaults."
        )


_SOURCE_SCHEMAS: dict[str, tuple[str, set[str]]] = {
    "demo": (
        "demo",
        {"adapter", "enabled", "live", "description"},
    ),
    "yahoo": (
        "yahoo_chart",
        {
            "adapter",
            "enabled",
            "live",
            "base_url",
            "timeout_seconds",
            "availability_time_utc",
        },
    ),
    "fred": (
        "fred",
        {
            "adapter",
            "enabled",
            "live",
            "base_url",
            "api_key_env",
            "timeout_seconds",
            "availability_time_utc",
        },
    ),
    "local_csv": (
        "csv",
        {"adapter", "enabled", "live", "description"},
    ),
}


def _validate_data_sources_config(payload: Mapping[str, Any]) -> None:
    """Validate the dispatcher-facing source registry and per-series blocks."""

    if not isinstance(payload, Mapping):
        raise ValueError("data_sources.yaml must contain a mapping")
    _reject_unknown_keys(
        payload,
        {"sources", "series"},
        "data_sources.yaml",
    )
    sources = payload.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("data_sources.yaml: `sources` must be a mapping")
    if not sources:
        raise ValueError("data_sources.yaml: at least one source provider is required")
    for provider, raw_settings in sources.items():
        provider_name = str(provider)
        if provider_name not in _SOURCE_SCHEMAS:
            raise ValueError(
                f"data_sources.yaml sources contains unsupported provider {provider_name!r}"
            )
        if not isinstance(raw_settings, Mapping):
            raise ValueError(
                f"data_sources.yaml sources.{provider_name} must be a mapping"
            )
        expected_adapter, allowed = _SOURCE_SCHEMAS[provider_name]
        _reject_unknown_keys(
            raw_settings,
            allowed,
            f"data_sources.yaml sources.{provider_name}",
        )
        adapter = raw_settings.get("adapter")
        if adapter is None:
            raise ValueError(
                f"data_sources.yaml sources.{provider_name}.adapter is required"
            )
        if str(adapter) != expected_adapter:
            raise ValueError(
                f"data_sources.yaml sources.{provider_name}.adapter must be "
                f"{expected_adapter!r}, not {str(adapter)!r}"
            )
        for flag in ("enabled", "live"):
            if flag in raw_settings and not isinstance(raw_settings[flag], bool):
                raise ValueError(
                    f"data_sources.yaml sources.{provider_name}.{flag} must be boolean"
                )
        if "timeout_seconds" in raw_settings:
            timeout = _finite_number(
                raw_settings["timeout_seconds"],
                f"data_sources.yaml sources.{provider_name}.timeout_seconds",
            )
            if timeout <= 0.0:
                raise ValueError(
                    f"data_sources.yaml sources.{provider_name}.timeout_seconds must be positive"
                )
        if "availability_time_utc" in raw_settings:
            try:
                availability_time = pd.to_timedelta(raw_settings["availability_time_utc"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"data_sources.yaml sources.{provider_name}.availability_time_utc must be HH:MM[:SS]"
                ) from exc
            if (
                availability_time < pd.Timedelta(0)
                or availability_time >= pd.Timedelta(days=1)
            ):
                raise ValueError(
                    f"data_sources.yaml sources.{provider_name}.availability_time_utc must fall within one UTC day"
                )

    series = payload.get("series", {})
    if not isinstance(series, Mapping):
        raise ValueError("data_sources.yaml: `series` must be a mapping")
    for series_id, raw_settings in series.items():
        if not isinstance(raw_settings, Mapping):
            raise ValueError(
                f"data_sources.yaml series.{series_id} must be a mapping"
            )
        provider = str(raw_settings.get("source", "demo"))
        registered_provider = "local_csv" if provider == "csv" else provider
        if registered_provider not in _SOURCE_SCHEMAS:
            raise ValueError(
                f"data_sources.yaml series.{series_id}.source references unsupported "
                f"provider {provider!r}"
            )
        if registered_provider not in sources:
            raise ValueError(
                f"data_sources.yaml series.{series_id}.source references unconfigured "
                f"provider {provider!r}"
            )


def _validate_catalysts_config(payload: Mapping[str, Any]) -> None:
    """Fail closed for the deterministic recurrence-calendar fixture."""

    if not isinstance(payload, Mapping):
        raise ValueError("catalysts.yaml must contain a mapping")
    _reject_unknown_keys(
        payload,
        {"calendar_type", "business_day_adjustment", "recurring"},
        "catalysts.yaml",
    )
    calendar_type = str(payload.get("calendar_type", "deterministic_fixture"))
    if calendar_type != "deterministic_fixture":
        raise ValueError(
            "catalysts.yaml recurrence rules must use "
            "calendar_type='deterministic_fixture'; generated dates cannot be "
            "relabelled as verified official releases"
        )
    adjustment = str(payload.get("business_day_adjustment", "following")).lower()
    if adjustment not in {"following", "none"}:
        raise ValueError(
            "catalysts.yaml business_day_adjustment must be 'following' or 'none'"
        )
    recurring = payload.get("recurring", [])
    if not isinstance(recurring, list):
        raise ValueError("catalysts.yaml recurring must be a list")

    from .regimes import REGIME_NAMES

    common = {
        "event",
        "cadence",
        "affected_regime",
        "expected_importance",
    }
    cadence_fields = {
        "monthly": {"day_of_month"},
        "weekly": {"weekday", "active_months"},
        "interval_days": {"anchor_date", "interval_days"},
    }
    valid_importance = {"low", "medium", "high", "critical"}
    valid_weekdays = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    }
    for position, raw_rule in enumerate(recurring):
        context = f"catalysts.yaml recurring[{position}]"
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"{context} must be a mapping")
        cadence = str(raw_rule.get("cadence", ""))
        if cadence not in cadence_fields:
            raise ValueError(f"{context}.cadence is unsupported: {cadence!r}")
        allowed = common | cadence_fields[cadence]
        _reject_unknown_keys(raw_rule, allowed, context)
        required = common | (
            {"day_of_month"}
            if cadence == "monthly"
            else {"weekday"}
            if cadence == "weekly"
            else {"anchor_date", "interval_days"}
        )
        missing = sorted(required.difference(raw_rule))
        if missing:
            raise ValueError(f"{context} is missing required key(s): {missing}")
        if not str(raw_rule["event"]).strip():
            raise ValueError(f"{context}.event cannot be empty")
        if str(raw_rule["affected_regime"]) not in REGIME_NAMES:
            raise ValueError(f"{context}.affected_regime is not a canonical regime")
        if str(raw_rule["expected_importance"]).lower() not in valid_importance:
            raise ValueError(f"{context}.expected_importance is unsupported")
        if cadence == "monthly":
            day = int(raw_rule["day_of_month"])
            if not 1 <= day <= 31:
                raise ValueError(f"{context}.day_of_month must be between 1 and 31")
        elif cadence == "weekly":
            if str(raw_rule["weekday"]).lower() not in valid_weekdays:
                raise ValueError(f"{context}.weekday is invalid")
            months = raw_rule.get("active_months", list(range(1, 13)))
            if not isinstance(months, list) or not months:
                raise ValueError(f"{context}.active_months must be a non-empty list")
            if any(int(month) not in range(1, 13) for month in months):
                raise ValueError(f"{context}.active_months must contain months 1..12")
        else:
            if int(raw_rule["interval_days"]) <= 0:
                raise ValueError(f"{context}.interval_days must be positive")
            try:
                pd.Timestamp(raw_rule["anchor_date"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.anchor_date must be a date") from exc


def _validate_operational_sections(main: Mapping[str, Any]) -> None:
    """Reject unused operational keys instead of silently accepting typos."""

    allowed: dict[str, set[str]] = {
        "system": {"name", "version"},
        "features": {
            "rolling_windows",
            "zscore_windows",
            "winsor_quantiles",
            "minimum_periods_fraction",
            "include_data_age",
        },
        "factors": {
            "normalization",
            "normalization_window",
            "min_periods",
            "history_lag",
            "winsor_limits",
            "z_clip",
            "wheat_momentum_window",
            "include_raw",
        },
        "backtest": {
            "annualization",
            "annualization_days",
            "execution_lag",
            "execution_lag_days",
            "signal_lag",
            "rebalance_frequency",
            "initial_capital",
            "transaction_cost_bps",
            "slippage_bps",
            "commission_bps",
            "futures_roll_cost_bps",
            "amortize_roll_cost",
            "risk_free_rate",
            "var_confidence",
            "apply_drawdown_control",
            "return_method",
            "roll_schedule",
            "annual_rolls",
            "futures_assets",
            "asset_aliases",
            "construct_portfolio",
            "static_long_asset",
            "momentum_lookback",
            "drawdown",
            "drawdown_controls",
        },
        "portfolio": {
            "annualization",
            "annualization_days",
            "vol_lookback",
            "min_volatility",
            "fallback_volatility",
            "target_volatility",
            "max_position_weight",
            "max_gross_exposure",
            "max_gross_leverage",
            "max_net_exposure",
            "max_asset_class_exposure",
            "max_asset_class_weight",
            "max_commodity_exposure",
            "max_commodity_weight",
            "commodity_asset_classes",
            "max_fertilizer_equity_exposure",
            "max_fertilizer_equity_weight",
            "max_factor_exposure",
            "confidence_floor",
            "min_signal",
            "min_return_observations",
            "require_latest_return",
            "covariance_is_annualized",
            "covariance_eigenvalue_floor",
            "asset_aliases",
            "asset_classes",
            "asset_class_limits",
            "factor_limits",
            "drawdown_controls",
        },
        "risk": {
            "signal_stop_threshold",
            "signal_stop_multiplier",
            "regime_stop_multiplier",
            "regime_stop_expectations",
        },
    }
    for section, section_allowed in allowed.items():
        values = main.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"config.yaml: `{section}` must be a mapping")
        _reject_unknown_keys(values, section_allowed, f"config.yaml {section}")

    models = main.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("config.yaml: `models` must be a mapping")
    _reject_unknown_keys(
        models,
        {"statistical_regime", "transitions"},
        "config.yaml models",
    )
    statistical_allowed = {
        "min_train_size",
        "min_class_observations",
        "refit_every",
        "l2",
        "max_iter",
        "learning_rate",
        "transparent_weight",
        "transparent_temperature",
        "classes",
    }
    transition_allowed = {
        "horizons",
        "smoothing",
        "min_train_size",
        "min_class_observations",
        "refit_every",
        "l2",
        "max_iter",
        "learning_rate",
        "empirical_weight",
        "states",
        "feature_columns",
    }
    for name, section_allowed in (
        ("statistical_regime", statistical_allowed),
        ("transitions", transition_allowed),
    ):
        values = models.get(name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"config.yaml models.{name} must be a mapping")
        _reject_unknown_keys(values, section_allowed, f"config.yaml models.{name}")


def _finite_number(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _validate_numeric_domains(
    main: Mapping[str, Any],
    runtime: RuntimeConfig,
    quality: QualityConfig,
) -> None:
    try:
        decision = pd.to_timedelta(runtime.decision_time_utc)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime.decision_time_utc must be HH:MM[:SS]") from exc
    if decision < pd.Timedelta(0) or decision >= pd.Timedelta(days=1):
        raise ValueError("runtime.decision_time_utc must fall within one UTC day")
    if int(runtime.warmup_business_days) < 0:
        raise ValueError("runtime.warmup_business_days cannot be negative")

    for name in ("warning_score", "failure_score"):
        value = _finite_number(getattr(quality, name), f"quality.{name}")
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"quality.{name} must be between 0 and 100")
    if quality.failure_score > quality.warning_score:
        raise ValueError("quality.failure_score cannot exceed warning_score")
    if not 0.0 <= float(quality.min_critical_coverage) <= 1.0:
        raise ValueError("quality.min_critical_coverage must be between 0 and 1")
    for name in (
        "abnormal_jump_z",
        "stale_penalty",
        "error_penalty",
        "warning_penalty",
    ):
        if _finite_number(getattr(quality, name), f"quality.{name}") < 0.0:
            raise ValueError(f"quality.{name} cannot be negative")

    features = dict(main.get("features", {}))
    for name in ("rolling_windows", "zscore_windows"):
        if any(int(value) <= 0 for value in features.get(name, [])):
            raise ValueError(f"features.{name} must contain positive integers")
    fraction = _finite_number(
        features.get("minimum_periods_fraction", 0.60),
        "features.minimum_periods_fraction",
    )
    if not 0.0 < fraction <= 1.0:
        raise ValueError("features.minimum_periods_fraction must be in (0, 1]")
    quantiles = list(features.get("winsor_quantiles", [0.01, 0.99]))
    if len(quantiles) != 2 or not 0.0 <= float(quantiles[0]) < float(quantiles[1]) <= 1.0:
        raise ValueError("features.winsor_quantiles must satisfy 0 <= low < high <= 1")

    factors = dict(main.get("factors", {}))
    for name in ("normalization_window", "min_periods", "wheat_momentum_window"):
        if int(factors.get(name, 1)) <= 0:
            raise ValueError(f"factors.{name} must be positive")
    if int(factors.get("history_lag", 0)) < 0:
        raise ValueError("factors.history_lag cannot be negative")
    factor_winsor = list(factors.get("winsor_limits", [0.01, 0.99]))
    if (
        len(factor_winsor) != 2
        or not 0.0
        <= float(factor_winsor[0])
        < float(factor_winsor[1])
        <= 1.0
    ):
        raise ValueError("factors.winsor_limits must satisfy 0 <= low < high <= 1")
    if _finite_number(factors.get("z_clip", 4.0), "factors.z_clip") <= 0.0:
        raise ValueError("factors.z_clip must be positive")

    backtest = dict(main.get("backtest", {}))
    annualization = backtest.get("annualization_days", backtest.get("annualization", 252))
    if _finite_number(annualization, "backtest.annualization") <= 0.0:
        raise ValueError("backtest annualization must be positive")
    if _finite_number(backtest.get("initial_capital", 1.0), "backtest.initial_capital") <= 0.0:
        raise ValueError("backtest.initial_capital must be positive")
    lag = backtest.get("execution_lag_days", backtest.get("execution_lag", 1))
    if int(lag) < 0:
        raise ValueError("backtest execution lag cannot be negative")
    rebalance = str(backtest.get("rebalance_frequency", "daily")).lower()
    if rebalance not in {"daily", "d", "business_daily", "b", "weekly", "w", "monthly", "m"}:
        raise ValueError("backtest.rebalance_frequency must be daily, weekly, or monthly")
    if str(backtest.get("return_method", "simple")).lower() not in {
        "simple",
        "pct",
        "percent",
        "log",
        "logarithmic",
    }:
        raise ValueError("backtest.return_method must be simple or log")
    if "var_confidence" in backtest:
        var_confidence = _finite_number(
            backtest["var_confidence"], "backtest.var_confidence"
        )
        if not 0.5 <= var_confidence < 1.0:
            raise ValueError("backtest.var_confidence must be in [0.5, 1)")
    for name in (
        "transaction_cost_bps",
        "slippage_bps",
        "commission_bps",
        "futures_roll_cost_bps",
    ):
        supplied = backtest.get(name, 0.0)
        values = supplied.values() if isinstance(supplied, Mapping) else [supplied]
        if any(_finite_number(value, f"backtest.{name}") < 0.0 for value in values):
            raise ValueError(f"backtest.{name} cannot be negative")

    portfolio = dict(main.get("portfolio", {}))
    for name in (
        "target_volatility",
        "max_position_weight",
        "max_gross_exposure",
        "max_gross_leverage",
        "max_net_exposure",
        "max_asset_class_exposure",
        "max_asset_class_weight",
        "max_commodity_exposure",
        "max_fertilizer_equity_exposure",
        "max_factor_exposure",
        "min_signal",
        "min_return_observations",
    ):
        if name in portfolio and _finite_number(portfolio[name], f"portfolio.{name}") < 0.0:
            raise ValueError(f"portfolio.{name} cannot be negative")
    confidence_floor = float(portfolio.get("confidence_floor", 0.0))
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("portfolio.confidence_floor must be between 0 and 1")
    for name in ("annualization", "annualization_days", "vol_lookback"):
        if (
            name in portfolio
            and _finite_number(portfolio[name], f"portfolio.{name}") <= 0.0
        ):
            raise ValueError(f"portfolio.{name} must be positive")
    for name in (
        "min_volatility",
        "fallback_volatility",
        "covariance_eigenvalue_floor",
    ):
        if (
            name in portfolio
            and _finite_number(portfolio[name], f"portfolio.{name}") <= 0.0
        ):
            raise ValueError(f"portfolio.{name} must be positive")
    for name in ("asset_class_limits", "factor_limits"):
        values = portfolio.get(name)
        if isinstance(values, Mapping) and any(
            _finite_number(value, f"portfolio.{name}") < 0.0
            for value in values.values()
        ):
            raise ValueError(f"portfolio.{name} values cannot be negative")
    controls = portfolio.get("drawdown_controls", [])
    if not isinstance(controls, list):
        raise ValueError("portfolio.drawdown_controls must be a list")
    for position, control in enumerate(controls):
        context = f"portfolio.drawdown_controls[{position}]"
        if not isinstance(control, Mapping):
            raise ValueError(f"{context} must be a mapping")
        _reject_unknown_keys(
            control,
            {"threshold", "risk_multiplier", "multiplier"},
            context,
        )
        threshold = _finite_number(control.get("threshold"), f"{context}.threshold")
        multiplier = _finite_number(
            control.get("risk_multiplier", control.get("multiplier")),
            f"{context}.risk_multiplier",
        )
        if threshold > 0.0:
            raise ValueError("drawdown-control thresholds must be non-positive")
        if not 0.0 <= multiplier <= 1.0:
            raise ValueError("drawdown-control multipliers must be between 0 and 1")

    risk = dict(main.get("risk", {}))
    for name in ("signal_stop_multiplier", "regime_stop_multiplier"):
        value = float(risk.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"risk.{name} must be between 0 and 1")
    if not 0.0 <= float(risk.get("signal_stop_threshold", 0.0)) <= 1.0:
        raise ValueError("risk.signal_stop_threshold must be between 0 and 1")

    models = dict(main.get("models", {}))
    for model_name in ("statistical_regime", "transitions"):
        model = dict(models.get(model_name, {}))
        for name in (
            "min_train_size",
            "min_class_observations",
            "refit_every",
            "max_iter",
        ):
            if name in model and int(model[name]) <= 0:
                raise ValueError(f"models.{model_name}.{name} must be positive")
        for name in ("l2", "smoothing"):
            if (
                name in model
                and _finite_number(model[name], f"models.{model_name}.{name}") < 0.0
            ):
                raise ValueError(f"models.{model_name}.{name} cannot be negative")
        if (
            "learning_rate" in model
            and _finite_number(
                model["learning_rate"], f"models.{model_name}.learning_rate"
            )
            <= 0.0
        ):
            raise ValueError(f"models.{model_name}.learning_rate must be positive")
        for name in ("transparent_weight", "empirical_weight"):
            if name in model:
                value = _finite_number(model[name], f"models.{model_name}.{name}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"models.{model_name}.{name} must be between 0 and 1"
                    )
    horizons = models.get("transitions", {}).get("horizons", {})
    if not isinstance(horizons, Mapping) or not horizons:
        raise ValueError("models.transitions.horizons must be a non-empty mapping")
    if any(int(value) <= 0 for value in horizons.values()):
        raise ValueError("models.transitions.horizons values must be positive")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    return payload


def _resolve_path(value: str | Path, base: Path) -> Path:
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "demo"
    start_date: str = "2015-01-01"
    end_date: str | None = None
    calendar: str = "B"
    decision_time_utc: str = "23:59:59"
    random_seed: int = 42
    strict_point_in_time: bool = True
    warmup_business_days: int = 504


@dataclass(frozen=True)
class QualityConfig:
    warning_score: float = 85.0
    failure_score: float = 60.0
    min_critical_coverage: float = 0.80
    abnormal_jump_z: float = 8.0
    stale_penalty: float = 4.0
    error_penalty: float = 12.0
    warning_penalty: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    """Complete immutable application configuration."""

    project_root: Path
    paths: PathConfig
    runtime: RuntimeConfig
    quality: QualityConfig
    series: dict[str, SeriesSpec]
    assets: dict[str, Any] = field(default_factory=dict)
    data_sources: dict[str, Any] = field(default_factory=dict)
    regime_thresholds: dict[str, Any] = field(default_factory=dict)
    catalysts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def series_spec(self, series_id: str) -> SeriesSpec:
        try:
            return self.series[series_id]
        except KeyError as exc:
            raise KeyError(f"Unknown canonical series: {series_id!r}") from exc


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load the main YAML plus referenced component files.

    Relative paths in the YAML are resolved from the project root rather than
    the current process directory, making scheduled runs reproducible.
    """

    main_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    project_root = main_path.parent.parent.resolve()
    main = _load_yaml(main_path)
    if overrides:
        main = _deep_merge(main, overrides)

    _reject_unknown_keys(
        main,
        {
            "system",
            "includes",
            "paths",
            "runtime",
            "quality",
            "features",
            "factors",
            "models",
            "backtest",
            "portfolio",
            "risk",
        },
        "config.yaml",
    )
    _validate_operational_sections(main)

    system = main.get("system", {})
    expected_name = "Agricultural Macro Quantitative Regime Monitor"
    from . import __version__

    if str(system.get("name", expected_name)) != expected_name:
        raise ValueError(
            "config.yaml system.name is an application identity, not a display override"
        )
    if str(system.get("version", __version__)) != __version__:
        raise ValueError(
            "config.yaml system.version must match the installed package version "
            f"{__version__!r}"
        )

    includes = main.get("includes", {})
    if not isinstance(includes, Mapping):
        raise ValueError("config.yaml: `includes` must be a mapping")
    _reject_unknown_keys(
        includes,
        {"assets", "data_sources", "regime_thresholds", "catalysts"},
        "config.yaml includes",
    )
    component_payloads: dict[str, dict[str, Any]] = {}
    for key in ("assets", "data_sources", "regime_thresholds", "catalysts"):
        include_path = includes.get(key, f"config/{key}.yaml")
        resolved = _resolve_path(include_path, project_root)
        component_payloads[key] = _load_yaml(resolved) if resolved.exists() else {}
    _validate_data_sources_config(component_payloads["data_sources"])
    _validate_catalysts_config(component_payloads["catalysts"])
    # Import locally to keep the configuration data classes independent of the
    # modelling module at import time while still failing on regime typos before
    # any ingestion or report output begins.
    from .regimes import regime_config_from_mapping

    regime_config_from_mapping(component_payloads["regime_thresholds"])

    path_values = main.get("paths", {})
    if not isinstance(path_values, Mapping):
        raise ValueError("config.yaml: `paths` must be a mapping")
    _reject_unknown_keys(
        path_values,
        {"data_dir", "raw_dir", "processed_dir", "cache_dir", "reports_dir"},
        "config.yaml paths",
    )
    data_dir = _resolve_path(path_values.get("data_dir", "data"), project_root)
    paths = PathConfig(
        data_dir=data_dir,
        raw_dir=_resolve_path(path_values.get("raw_dir", data_dir / "raw"), project_root),
        processed_dir=_resolve_path(path_values.get("processed_dir", data_dir / "processed"), project_root),
        cache_dir=_resolve_path(path_values.get("cache_dir", data_dir / "cache"), project_root),
        reports_dir=_resolve_path(path_values.get("reports_dir", "reports"), project_root),
    )

    runtime_values = main.get("runtime", {})
    if not isinstance(runtime_values, Mapping):
        raise ValueError("config.yaml: `runtime` must be a mapping")
    _reject_unknown_keys(
        runtime_values,
        set(RuntimeConfig.__dataclass_fields__),
        "config.yaml runtime",
    )
    runtime = RuntimeConfig(
        **dict(runtime_values)
    )
    if runtime.mode not in {"demo", "live", "cache"}:
        raise ValueError("runtime.mode must be one of: demo, live, cache")
    if int(runtime.warmup_business_days) < 0:
        raise ValueError("runtime.warmup_business_days cannot be negative")
    quality_values = main.get("quality", {})
    if not isinstance(quality_values, Mapping):
        raise ValueError("config.yaml: `quality` must be a mapping")
    _reject_unknown_keys(
        quality_values,
        set(QualityConfig.__dataclass_fields__),
        "config.yaml quality",
    )
    quality = QualityConfig(
        **dict(quality_values)
    )
    _validate_numeric_domains(main, runtime, quality)

    series_payload = component_payloads["data_sources"].get("series", {})
    series = {
        str(series_id): SeriesSpec.from_mapping(str(series_id), values or {})
        for series_id, values in series_payload.items()
    }
    if not series:
        raise ValueError("At least one series must be configured in data_sources.yaml")

    reproducible_raw = dict(main)
    reproducible_raw["resolved_includes"] = component_payloads
    return AppConfig(
        project_root=project_root,
        paths=paths,
        runtime=runtime,
        quality=quality,
        series=series,
        assets=component_payloads["assets"],
        data_sources=component_payloads["data_sources"],
        regime_thresholds=component_payloads["regime_thresholds"],
        catalysts=component_payloads["catalysts"],
        raw=reproducible_raw,
    )
