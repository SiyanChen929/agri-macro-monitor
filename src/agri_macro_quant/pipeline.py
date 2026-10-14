"""End-to-end point-in-time agricultural macro research workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from . import __version__
from .alignment import align_point_in_time
from .backtest import (
    BacktestResult,
    compare_benchmarks,
    run_backtest,
    slice_backtest_result,
)
from .config import AppConfig, load_config
from .economics import build_economic_features
from .factors import build_mvp_factors, factor_coverage
from .features import build_features
from .ingestion import build_catalyst_calendar, ingest_all, save_snapshot
from .monitoring import model_health_report
from .portfolio import (
    aggregate_executable_signals,
    construct_portfolio,
    construct_portfolio_history,
)
from .provenance import (
    build_manifest,
    capture_reproducibility_state,
    file_hash,
    stable_json_hash,
    write_manifest,
)
from .regimes import (
    RegimeResult,
    StatisticalRegimeResult,
    estimate_statistical_regimes,
    regime_config_from_mapping,
    score_regimes,
)
from .reporting import generate_reports
from .research import ASSOCIATION_DISCLAIMER, LeadLagAnalysisResult, run_lead_lag_analysis
from .risk import apply_drawdown_control, apply_signal_regime_stops, portfolio_risk_report
from .scenarios import historical_stress_test, run_scenario_analysis
from .signals import generate_relative_value_signals, generate_trade_signals
from .transitions import TransitionResult, estimate_transition_probabilities


REGIME_SIGNAL_ALIASES = {
    "weather_crop_supply_shock": "weather_shock",
    "inventory_food_security_stress": "food_security",
    "government_trade_policy_amplification": "policy_stress",
    "agricultural_input_expansion": "input_expansion",
    "food_inflation_transmission": "food_inflation",
    "monetary_tightening_stagflation": "monetary_tightening",
    "supply_response_mean_reversion": "supply_response",
}

MARKET_TO_ASSET = {
    "wheat": "ZW",
    "corn": "ZC",
    "soybeans": "ZS",
    "natural_gas": "NG",
    "cf": "CF",
    "ntr": "NTR",
    "mos": "MOS",
    "dxy": "DXY",
}

FACTOR_EXPOSURES = {
    "ZW": {"grain": 1.0, "ag_inflation": 0.9, "policy_risk": 0.4, "usd": -0.2},
    "ZC": {"grain": 0.9, "ag_inflation": 0.8, "policy_risk": 0.25, "usd": -0.2},
    "ZS": {"grain": 0.75, "ag_inflation": 0.65, "policy_risk": 0.25, "usd": -0.2},
    "NG": {"energy": 1.0, "ag_inflation": 0.25, "usd": -0.15},
    "CF": {"fertilizer": 1.0, "grain": 0.35, "energy": -0.45, "equity_beta": 0.55},
    "NTR": {"fertilizer": 0.9, "grain": 0.35, "policy_risk": 0.25, "equity_beta": 0.55},
    "MOS": {"fertilizer": 0.85, "grain": 0.30, "policy_risk": 0.25, "equity_beta": 0.60},
    "DXY": {"usd": 1.0, "rates": 0.35, "ag_inflation": -0.2},
    "US2Y": {"rates": 1.0, "ag_inflation": 0.25, "usd": 0.2},
}

LEAD_LAG_RELATIONSHIPS = (
    {
        "label": "Q1 weather stress -> wheat",
        "dependent": "wheat",
        "predictor": "weather_stress",
    },
    {
        "label": "Q5 corn/urea -> CF",
        "dependent": "cf",
        "predictor": "corn_urea_raw",
    },
    {
        "label": "Q6 wheat -> breakeven inflation",
        "dependent": "breakeven_5y",
        "predictor": "wheat",
    },
    {
        "label": "Q7 breakevens -> US 2Y",
        "dependent": "us_2y",
        "predictor": "breakeven_5y",
    },
    {
        "label": "Q8 wheat -> CF",
        "dependent": "cf",
        "predictor": "wheat",
    },
)


@dataclass
class PipelineResult:
    output_dir: Path
    observations: pd.DataFrame
    market: pd.DataFrame
    features: pd.DataFrame
    factors: pd.DataFrame
    signal_factors: pd.DataFrame
    regimes: RegimeResult
    statistical_regimes: StatisticalRegimeResult | None
    transitions: TransitionResult
    signals: pd.DataFrame
    portfolio_weights: pd.Series
    backtest: BacktestResult
    risk: dict[str, Any]
    scenarios: pd.DataFrame
    stress_tests: pd.DataFrame
    benchmarks: pd.DataFrame
    catalysts: pd.DataFrame
    model_health: dict[str, Any]
    lead_lag: LeadLagAnalysisResult
    research_questions: pd.DataFrame
    quality: Any
    manifest: dict[str, Any]
    reports: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        if hasattr(value, "as_dict"):
            return _jsonable(value.as_dict())
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        return None if isinstance(item, float) and not np.isfinite(item) else item
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _configuration_snapshot(
    config: AppConfig,
    effective_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = {
        "main": config.raw,
        "assets": config.assets,
        "data_sources": config.data_sources,
        "regime_thresholds": config.regime_thresholds,
        "catalysts": config.catalysts,
        "resolved_paths": {key: str(value) for key, value in asdict(config.paths).items()},
    }
    if effective_runtime is not None:
        snapshot["effective_runtime"] = dict(effective_runtime)
    return snapshot


def _feature_configuration(config: AppConfig) -> dict[str, Any]:
    raw = dict(config.raw.get("features", {}))
    return {
        "include_original": True,
        "changes": raw.get("rolling_windows", [5, 20, 60, 120]),
        "zscore_windows": raw.get("zscore_windows", [60, 252]),
        "percentile_windows": [252],
        "momentum_windows": [20, 60],
        "volatility_windows": [20],
        "trend_windows": [20, 60],
        "winsor_limits": tuple(raw.get("winsor_quantiles", [0.01, 0.99])),
        "minimum_periods_fraction": float(raw.get("minimum_periods_fraction", 0.60)),
        "robust": True,
        "z_clip": 5.0,
        # The aligned frame is already publication-lag controlled.  Current
        # values are safe to use for an end-of-day signal; execution is lagged.
        "history_lag": 0,
    }


def _factor_configuration(config: AppConfig) -> dict[str, Any]:
    raw = dict(config.raw.get("factors", {}))
    return {
        "normalization": raw.get("normalization", "robust_zscore"),
        "normalization_window": int(raw.get("normalization_window", 252)),
        "min_periods": int(raw.get("min_periods", 20)),
        "history_lag": int(raw.get("history_lag", 0)),
        "winsor_limits": tuple(raw.get("winsor_limits", [0.01, 0.99])),
        "z_clip": float(raw.get("z_clip", 4.0)),
        "wheat_momentum_window": int(raw.get("wheat_momentum_window", 60)),
        "include_raw": bool(raw.get("include_raw", False)),
    }


def _model_configuration(config: AppConfig, key: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    configured = config.raw.get("models", {}).get(key, {})
    result = dict(defaults)
    if isinstance(configured, Mapping):
        result.update(dict(configured))
    result["random_seed"] = config.runtime.random_seed
    return result


def _merge_signal_factors(mvp: pd.DataFrame, economics: pd.DataFrame) -> pd.DataFrame:
    result = mvp.copy()
    for column in economics.columns:
        if column not in result.columns:
            result[column] = economics[column]
    return result.sort_index()


def _lead_lag_research(frame: pd.DataFrame) -> LeadLagAnalysisResult:
    eligible: list[dict[str, str]] = []
    blocked: list[dict[str, Any]] = []
    for definition in LEAD_LAG_RELATIONSHIPS:
        dependent = definition["dependent"]
        predictor = definition["predictor"]
        complete = (
            frame[[dependent, predictor]].apply(pd.to_numeric, errors="coerce").diff().dropna()
            if {dependent, predictor}.issubset(frame.columns)
            else pd.DataFrame()
        )
        if len(complete) >= 30:
            eligible.append(dict(definition))
        else:
            blocked.append(
                {
                    "relationship": definition["label"],
                    "dependent": dependent,
                    "predictor": predictor,
                    "transform": "differences",
                    "complete_pair_observations": int(len(complete)),
                    "status": "not_tested_insufficient_data",
                    "association_not_causality": True,
                }
            )
    if eligible:
        result = run_lead_lag_analysis(
            frame,
            relationships=eligible,
            max_lag=5,
            transform="differences",
        )
        result.metadata["status"] = "tested"
    else:
        result = LeadLagAnalysisResult(
            cross_correlations=pd.DataFrame(),
            distributed_lag_coefficients=pd.DataFrame(),
            granger_tests=pd.DataFrame(),
            metadata=pd.DataFrame(),
            disclaimer=ASSOCIATION_DISCLAIMER,
        )
    if blocked:
        result.metadata = pd.concat(
            [result.metadata, pd.DataFrame(blocked)],
            ignore_index=True,
            sort=False,
        )
    return result


def _research_question_status(
    lead_lag: LeadLagAnalysisResult,
    transitions: TransitionResult,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    tested = set(
        lead_lag.metadata.loc[
            lead_lag.metadata.get("status", pd.Series(index=lead_lag.metadata.index, dtype=str))
            .astype(str)
            .eq("tested"),
            "relationship",
        ].astype(str)
    ) if not lead_lag.metadata.empty and "relationship" in lead_lag.metadata else set()
    rows = [
        ("Q1", "tested" if "Q1 weather stress -> wheat" in tested else "not_tested", "lead/lag and Granger diagnostics", "lead_lag_granger_tests.csv"),
        ("Q2", "not_tested_missing_global_stocks", "exporter versus global stocks predictor comparison", "requires a PIT global stocks-to-use series"),
        ("Q3", "not_tested_missing_event_history", "event-study abnormal returns", "requires dated export-restriction events"),
        ("Q4", "partially_tested", "physical-scarcity interaction enters signals", "standalone interaction benchmark not yet estimated"),
        ("Q5", "tested" if "Q5 corn/urea -> CF" in tested else "not_tested", "lead/lag and Granger diagnostics", "lead_lag_granger_tests.csv"),
        ("Q6", "tested" if "Q6 wheat -> breakeven inflation" in tested else "not_tested", "lead/lag and Granger diagnostics", "lead_lag_granger_tests.csv"),
        (
            "Q7",
            "proxy_tested" if "Q7 breakevens -> US 2Y" in tested else "not_tested",
            "lead/lag diagnostics using US 2Y as a Fed-repricing proxy",
            "does not observe meeting-level policy-expectation repricing",
        ),
        (
            "Q8",
            "partially_tested" if "Q8 wheat -> CF" in tested else "not_tested",
            "wheat-change to CF-change lead/lag diagnostics",
            "predictive precedence is not a direct test of relative peak dates",
        ),
        (
            "Q9",
            "partially_tested"
            if any(not table.empty for table in transitions.model_coefficients.values())
            else "empirical_fallback",
            "walk-forward conditional transition model",
            "coefficients exist, but no nested variable-importance comparison is claimed",
        ),
        ("Q10", "tested" if "time_series_momentum" in benchmarks.index else "not_tested", "cost-aware regime strategy versus momentum benchmark", "benchmark_comparison.csv"),
    ]
    return pd.DataFrame(rows, columns=["question", "status", "method", "evidence_or_limitation"])


def _traded_prices(market: pd.DataFrame) -> pd.DataFrame:
    columns = {source: target for source, target in MARKET_TO_ASSET.items() if source in market}
    return market[list(columns)].rename(columns=columns).apply(pd.to_numeric, errors="coerce")


def _traded_returns(
    market: pd.DataFrame,
    observation_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prices = _traded_prices(market)
    valid_prices = prices.where(prices > 0.0)
    # Preserve the full move to the next valid mark while retaining NaN on the
    # unpriced dates themselves.  Passing a precomputed pct_change across a gap
    # would otherwise bypass the backtester's sparse-price protection and lose
    # the cumulative move permanently.
    output = valid_prices.ffill().pct_change(fill_method=None).where(
        valid_prices.notna()
    )
    if observation_dates is not None:
        source_columns = {
            source: target
            for source, target in MARKET_TO_ASSET.items()
            if source in observation_dates.columns and target in output.columns
        }
        observed = observation_dates[list(source_columns)].rename(
            columns=source_columns
        ).reindex(output.index)
        fresh = pd.DataFrame(False, index=output.index, columns=output.columns)
        decision_dates = pd.Series(output.index.normalize(), index=output.index)
        for asset in observed.columns:
            actual_dates = pd.to_datetime(observed[asset], errors="coerce").dt.normalize()
            fresh[asset] = actual_dates.eq(decision_dates)
        # Carried marks remain useful for valuation and make the next observed
        # return a full catch-up move, but are not executable trading marks.
        output = output.where(fresh)
    return output.replace([np.inf, -np.inf], np.nan)


def _portfolio_configuration(config: AppConfig) -> dict[str, Any]:
    raw = dict(config.raw.get("portfolio", {}))
    raw.setdefault("target_volatility", 0.10)
    raw.setdefault("max_net_exposure", 0.35)
    raw.setdefault("max_commodity_exposure", 0.40)
    raw.setdefault("max_fertilizer_equity_exposure", 0.35)
    raw.setdefault("max_factor_exposure", 0.45)
    raw.setdefault("asset_classes", {})
    return raw


def _backtest_configuration(config: AppConfig, portfolio_config: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(config.raw.get("backtest", {}))
    raw["portfolio"] = dict(portfolio_config)
    raw["factor_exposures"] = FACTOR_EXPOSURES
    raw.setdefault("futures_assets", ["ZW", "ZC", "ZS", "NG"])
    portfolio_raw = config.raw.get("portfolio", {})
    if "drawdown_controls" in portfolio_raw:
        raw["drawdown_controls"] = portfolio_raw["drawdown_controls"]
    return raw


def _point_in_time_quality_multiplier(
    data: pd.DataFrame,
    age_days: pd.DataFrame,
    specs: Mapping[str, Any],
) -> pd.Series:
    """Return a causal data-availability haircut for every decision date.

    The run-level :class:`DataQualityReport` is intentionally *not* used here:
    it is evaluated at the run endpoint and therefore cannot be projected
    backwards over a historical backtest.  Instead, each date is scored from
    the configured critical series that were both present and within their
    staleness limit on that date.  The resulting critical-series coverage is
    intrinsically prefix invariant because no expanding/full-sample statistic
    enters the calculation.

    ``PITAligner`` already masks stale values.  Rechecking the age panel here
    makes the contract explicit and fails conservatively if a value and its
    age metadata disagree.
    """

    if not isinstance(data, pd.DataFrame) or not isinstance(age_days, pd.DataFrame):
        raise TypeError("data and age_days must be pandas DataFrames")
    if not data.index.equals(age_days.index):
        raise ValueError("data and age_days must use the same decision-date index")

    critical = [
        str(series_id)
        for series_id, spec in specs.items()
        if bool(getattr(spec, "critical", False))
    ]
    if not critical:
        return pd.Series(1.0, index=data.index, name="point_in_time_quality_multiplier")

    usable = pd.DataFrame(False, index=data.index, columns=critical, dtype=bool)
    for series_id in critical:
        if series_id not in data.columns or series_id not in age_days.columns:
            continue
        spec = specs[series_id]
        age = pd.to_numeric(age_days[series_id], errors="coerce")
        valid = data[series_id].notna() & age.notna() & age.ge(0.0)
        staleness_limit = getattr(spec, "staleness_limit", None)
        if staleness_limit is not None:
            valid &= age.le(float(staleness_limit))
        usable[series_id] = valid

    multiplier = usable.mean(axis=1).astype(float).clip(0.0, 1.0)
    multiplier.name = "point_in_time_quality_multiplier"
    return multiplier


def _quarantine_invalid_aligned_values(
    data: pd.DataFrame,
    specs: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causally mask nonfinite, out-of-bounds, and implausible-jump values.

    Validation reports remain non-mutating evidence about the source feed, but
    invalid finite observations must not enter factors merely because they are
    technically available. Return-limit checks compare each row only with the
    latest prior valid value; rejected rows never contaminate the reference for
    later decisions.
    """

    clean = data.copy()
    invalid = pd.DataFrame(False, index=data.index, columns=data.columns)
    for series_id, spec in specs.items():
        if series_id not in clean.columns:
            continue
        values = pd.to_numeric(clean[series_id], errors="coerce")
        rejected = values.notna() & ~np.isfinite(values)
        minimum = getattr(spec, "min_value", None)
        maximum = getattr(spec, "max_value", None)
        if minimum is not None:
            rejected |= values.lt(float(minimum))
        if maximum is not None:
            rejected |= values.gt(float(maximum))

        return_limit = getattr(spec, "max_abs_daily_return", None)
        if return_limit is not None:
            last_valid: float | None = None
            for position, value in enumerate(values.to_numpy(dtype=float)):
                if not np.isfinite(value) or bool(rejected.iloc[position]):
                    continue
                if last_valid is not None:
                    if abs(last_valid) <= 1e-12:
                        jump = abs(value - last_valid) > 1e-12
                    else:
                        jump = abs(value / last_valid - 1.0) > float(return_limit)
                    if jump:
                        rejected.iloc[position] = True
                        continue
                last_valid = float(value)

        invalid[series_id] = rejected.fillna(False)
        clean.loc[invalid[series_id], series_id] = np.nan
    clean.attrs.update(data.attrs)
    invalid.index.name = data.index.name
    return clean, invalid


def _apply_point_in_time_confidence(
    signals: pd.DataFrame,
    data: pd.DataFrame,
    age_days: pd.DataFrame,
    specs: Mapping[str, Any],
    *,
    final_quality_multiplier: float,
) -> pd.DataFrame:
    """Attach causal backtest confidence and a current-only quality haircut.

    ``backtest_confidence`` depends only on the signal row and point-in-time
    critical-series coverage on that row's date.  ``confidence`` additionally
    applies the final run-level quality multiplier, but only to rows on the
    latest signal date.  Current portfolio construction uses ``confidence``;
    the historical backtest must use ``backtest_confidence``.
    """

    result = signals.copy()
    raw = pd.to_numeric(result.get("confidence"), errors="coerce").fillna(0.0)
    result["raw_confidence"] = raw.clip(0.0, 100.0)

    date_multiplier = _point_in_time_quality_multiplier(data, age_days, specs)
    signal_dates = pd.to_datetime(result.get("date"), errors="coerce")
    multiplier = signal_dates.map(date_multiplier).fillna(0.0).astype(float)
    result["point_in_time_quality_multiplier"] = multiplier.clip(0.0, 1.0)
    result["backtest_confidence"] = (
        result["raw_confidence"] * result["point_in_time_quality_multiplier"]
    ).clip(0.0, 100.0)

    result["current_quality_multiplier"] = 1.0
    if signal_dates.notna().any():
        latest = signal_dates.max()
        current_multiplier = float(np.clip(final_quality_multiplier, 0.0, 1.0))
        result.loc[signal_dates.eq(latest), "current_quality_multiplier"] = current_multiplier
    result["confidence"] = (
        result["backtest_confidence"] * result["current_quality_multiplier"]
    ).clip(0.0, 100.0)
    return result


def _apply_asset_tradability_gate(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    min_return_observations: int = 20,
) -> pd.DataFrame:
    """Zero confidence when an absolute asset or either pair leg is unpriced.

    The check is date-local and causal: a row needs a valid return/mark on its
    own decision date and the configured number of observations no later than
    that date. Pair rows require both executable legs.
    """

    result = signals.copy()
    if result.empty:
        result["tradable"] = pd.Series(dtype=bool)
        result["tradability_reason"] = pd.Series(dtype=str)
        return result
    history = returns.apply(pd.to_numeric, errors="coerce").sort_index()
    counts = history.notna().cumsum()
    required = max(1, int(min_return_observations))

    def check(asset: Any, date: pd.Timestamp) -> str | None:
        name = str(asset)
        if name not in history.columns:
            return f"{name}: no return series"
        if date not in history.index:
            return f"{name}: decision date not in return calendar"
        observed = int(counts.at[date, name])
        if observed < required:
            return f"{name}: only {observed} valid returns; {required} required"
        if pd.isna(history.at[date, name]):
            return f"{name}: latest return/mark unavailable"
        return None

    dates = pd.to_datetime(result["date"], errors="coerce")
    reasons: list[str] = []
    for position, (_, row) in enumerate(result.iterrows()):
        date = dates.iloc[position]
        if pd.isna(date):
            reasons.append("invalid decision date")
            continue
        if pd.notna(row.get("long_leg")) and pd.notna(row.get("short_leg")):
            legs = (row.get("long_leg"), row.get("short_leg"))
        else:
            legs = (row.get("asset"),)
        failures = [reason for leg in legs if (reason := check(leg, date))]
        reasons.append("; ".join(failures))

    result["tradability_reason"] = reasons
    result["tradable"] = result["tradability_reason"].eq("")
    for column in ("confidence", "backtest_confidence"):
        if column in result:
            result.loc[~result["tradable"], column] = 0.0
    return result


def _write_csv(frame: pd.DataFrame | pd.Series, path: Path, *, index: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(path, index=index, compression=compression)
    return path


def _write_json(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


def _prepare_output_dir(
    config: AppConfig,
    output_dir: str | Path | None,
    *,
    overwrite: bool,
) -> Path:
    marker_name = ".agri_macro_quant_run.json"

    def write_marker(path: Path) -> None:
        marker = {
            "format": "agri_macro_quant_run_v1",
            "project_root": str(config.project_root.resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (path / marker_name).write_text(
            json.dumps(marker, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    def is_managed_run(path: Path) -> bool:
        marker = path / marker_name
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return payload.get("format") == "agri_macro_quant_run_v1"
        # Backward compatibility for completed v0.1 run directories created
        # before the explicit marker was introduced.
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return bool(
                manifest.get("system_version")
                and manifest.get("data_snapshot_sha256")
                and manifest.get("code_revision")
            )
        return False

    if output_dir is None:
        reports_root = config.paths.reports_dir.expanduser().resolve()
        reports_root.mkdir(parents=True, exist_ok=True)
        destination: Path | None = None
        for _ in range(10):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            candidate = reports_root / f"run_{stamp}_{uuid4().hex[:8]}"
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:  # practically impossible, but atomic
                continue
            destination = candidate
            break
        if destination is None:
            raise FileExistsError("Could not allocate a unique run directory")
        write_marker(destination)
        return destination
    else:
        destination = Path(output_dir).expanduser()
    if destination.is_symlink():
        raise ValueError(f"Refusing to use a symbolic link as an output directory: {destination}")
    destination = destination.resolve()

    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {destination}")

    children = list(destination.iterdir()) if destination.exists() else []
    if children and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Pass overwrite=True to replace the complete run directory."
        )
    if children and overwrite:
        # --overwrite means a fresh artifact set, not a merge.  Validate the
        # exact resolved target before removing only its immediate contents so
        # capability-specific artifacts from a prior run cannot survive.
        protected = {
            Path(destination.anchor).resolve(),
            Path.home().resolve(),
            Path(tempfile.gettempdir()).resolve(),
            config.project_root.resolve(),
        }
        for configured_path in asdict(config.paths).values():
            protected.add(Path(configured_path).expanduser().resolve())
        if destination in protected or destination in config.project_root.resolve().parents:
            raise ValueError(
                f"Refusing to overwrite protected or broad directory: {destination}. "
                "Choose a dedicated run subdirectory."
            )
        if not is_managed_run(destination):
            raise ValueError(
                f"Refusing to overwrite an unmarked directory: {destination}. "
                "Only a prior agri-macro run directory containing its run marker "
                "or a valid manifest may be replaced."
            )
        for child in children:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                raise ValueError(f"Unsupported output entry cannot be safely replaced: {child}")
    destination.mkdir(parents=True, exist_ok=True)
    write_marker(destination)
    return destination


def _source_label(mode: str, observation_attrs: Mapping[str, Any] | None = None) -> str:
    attrs = dict(observation_attrs or {})
    origin_mode = str(attrs.get("origin_mode", attrs.get("mode", mode)))
    synthetic = bool(attrs.get("synthetic", origin_mode == "demo"))
    if synthetic or origin_mode == "demo":
        if mode == "cache":
            return (
                "DEMO-ORIGIN VERIFIED CACHE — deterministic synthetic point-in-time "
                "fixture; not investment data"
            )
        return "DEMO — deterministic synthetic point-in-time fixture; not investment data"
    if mode == "cache":
        return "Verified canonical cache snapshot"
    return "Configured live sources (public or user-supplied); unavailable series are not silently synthesized"


def run_pipeline(
    *,
    config_path: str | Path | None = None,
    mode: str | None = None,
    start: Any | None = None,
    as_of: Any | None = None,
    cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    fail_on_quality: bool = False,
    run_statistical: bool = True,
    record_research_log: bool = True,
) -> PipelineResult:
    """Execute ingestion → measurement → signals → portfolio → report.

    ``demo`` is the default and needs neither network access nor credentials.
    ``live`` never fills unavailable fundamental feeds with synthetic values;
    their absence becomes a quality failure and confidence/positions fall to
    zero unless the configuration supplies real adapters.
    """

    config = load_config(config_path)
    reproducibility_state = capture_reproducibility_state(config.project_root)
    if not config.runtime.strict_point_in_time:
        raise ValueError(
            "runtime.strict_point_in_time=false is unsupported: this system always enforces publication-time alignment"
        )
    selected_mode = str(mode or config.runtime.mode)
    start_date = pd.Timestamp(start or config.runtime.start_date).normalize()
    as_of_date = pd.Timestamp(as_of or config.runtime.end_date or pd.Timestamp.now()).normalize()
    warmup_business_days = max(0, int(config.runtime.warmup_business_days))
    fetch_start = start_date - pd.offsets.BDay(warmup_business_days)
    destination = _prepare_output_dir(config, output_dir, overwrite=overwrite)

    observations = ingest_all(
        config,
        selected_mode,
        start=fetch_start,
        end=as_of_date,
        cache_path=cache_path,
    )
    observation_attrs = dict(observations.attrs)
    aligned = align_point_in_time(
        observations,
        config.series,
        start=fetch_start,
        end=as_of_date,
        as_of=as_of_date,
        decision_time_utc=config.runtime.decision_time_utc,
        calendar_frequency=config.runtime.calendar,
        quality_config=asdict(config.quality),
    )
    quality = aligned.quality
    if fail_on_quality and not quality.passed:
        _write_json(quality.as_dict(), destination / "quality_failure.json")
        raise RuntimeError(f"DATA QUALITY WARNING: validation status is {quality.status}")

    market, invalid_value_mask = _quarantine_invalid_aligned_values(
        aligned.data,
        config.series,
    )
    features = build_features(market, _feature_configuration(config))
    if bool(config.raw.get("features", {}).get("include_data_age", True)):
        features = features.join(aligned.age_days.add_suffix("__data_age_days"))
    factors = build_mvp_factors(market, _factor_configuration(config))
    economic_features = build_economic_features(market, normalization_window=252, min_periods=20)
    signal_factors = _merge_signal_factors(factors, economic_features)
    research_frame = market.join(
        economic_features[["corn_urea_raw"]]
        if "corn_urea_raw" in economic_features
        else pd.DataFrame(index=market.index),
        how="left",
    )
    lead_lag = _lead_lag_research(research_frame)

    regime_config = regime_config_from_mapping(config.regime_thresholds)
    regimes = score_regimes(factors, regime_config)
    latest_regime_scores = regimes.scores.iloc[-1].dropna() if not regimes.scores.empty else pd.Series(dtype=float)
    if regimes.scores.dropna(how="all").empty or latest_regime_scores.empty:
        failure = quality.as_dict()
        failure["pipeline_error"] = (
            "No point-in-time regime score is available on the latest decision date; "
            "signals and positions were not produced."
        )
        _write_json(failure, destination / "quality_failure.json")
        raise RuntimeError(
            "DATA QUALITY WARNING: no regime can be scored on the latest decision date"
        )
    statistical: StatisticalRegimeResult | None = None
    if run_statistical:
        statistical = estimate_statistical_regimes(
            factors,
            labels=regimes.dominant_regime,
            transparent=regimes,
            config=_model_configuration(config, "statistical_regime", {
                "min_train_size": 252,
                "min_class_observations": 10,
                "refit_every": 63,
                "max_iter": 180,
            }),
        )
    transitions = estimate_transition_probabilities(
        regimes,
        features=factors,
        config=_model_configuration(config, "transitions", {
            "horizons": {"1w": 5, "1m": 21, "3m": 63},
            "min_train_size": 252,
            "min_class_observations": 10,
            "refit_every": 63,
            "max_iter": 160,
        }),
    )

    signal_regimes = regimes.scores.rename(columns=REGIME_SIGNAL_ALIASES)
    transition_history = transitions.forecast_history.get("1m")
    absolute_signals = generate_trade_signals(
        signal_factors,
        signal_regimes,
        transition_probs=transition_history,
    )
    prices = _traded_prices(market)
    relative_signals = generate_relative_value_signals(
        prices,
        regime_scores=signal_regimes,
        config={"lookback": 60, "min_periods": 20},
    )
    signals = pd.concat([absolute_signals, relative_signals], ignore_index=True, sort=False)
    quality_multiplier = float(quality.confidence_multiplier)
    signals = _apply_point_in_time_confidence(
        signals,
        market,
        aligned.age_days,
        config.series,
        final_quality_multiplier=quality_multiplier,
    )
    returns = _traded_returns(market, aligned.observation_dates)
    signals = _apply_asset_tradability_gate(
        signals,
        returns,
        min_return_observations=int(
            config.raw.get("portfolio", {}).get("min_return_observations", 20)
        ),
    )
    if observation_attrs.get("synthetic"):
        signals["data_mode"] = "demo_synthetic"
    else:
        signals["data_mode"] = selected_mode

    portfolio_config = _portfolio_configuration(config)
    factor_exposures = pd.DataFrame.from_dict(FACTOR_EXPOSURES, orient="index")
    uncontrolled_weights = construct_portfolio(
        signals,
        returns=returns,
        factor_exposures=factor_exposures,
        config=portfolio_config,
    )
    backtest_config = _backtest_configuration(config, portfolio_config)
    # The endpoint quality report can legitimately change when later data are
    # appended, so it must never enter historical target weights.  Keep the
    # current-only confidence in ``signals`` for portfolio/reporting, while
    # the backtest consumes the date-local, prefix-invariant field.
    historical_signals = signals.copy()
    historical_signals["confidence"] = historical_signals["backtest_confidence"]
    target_history = construct_portfolio_history(
        historical_signals,
        returns,
        factor_exposures=factor_exposures,
        config=portfolio_config,
    )
    # Missing/untradable is a hold/no-event state, not an instruction to
    # flatten. Explicit zeros from valid evidence remain executable zeros.
    target_mark_availability = returns.notna().reindex(
        index=target_history.index,
        columns=target_history.columns,
        fill_value=False,
    )
    target_history = target_history.where(target_mark_availability).dropna(how="all")
    # Run the accounting state machine across the complete causal warm-up.
    # Retaining only an execution-lag-sized prefix would lose drifted holdings,
    # pending fills, and the last weekly/monthly target at the evaluation
    # boundary, fabricating start-date turnover and costs.
    warmup_backtest = run_backtest(
        returns=returns,
        target_weights=target_history,
        config=backtest_config,
    )
    backtest = slice_backtest_result(
        warmup_backtest,
        start_date,
        config=backtest_config,
        asset_returns=returns,
    )
    risk_config = dict(config.raw.get("risk", {}))
    current_signal_vector = aggregate_executable_signals(signals)["signal"]
    weights = apply_signal_regime_stops(
        uncontrolled_weights,
        signals=current_signal_vector,
        current_regimes=str(regimes.dominant_regime.iloc[-1]),
        expected_regimes=risk_config.get("regime_stop_expectations"),
        config=risk_config,
    )
    drawdown_config = {
        **risk_config,
        "input_type": "equity",
        "drawdown_controls": config.raw.get("portfolio", {}).get(
            "drawdown_controls", []
        ),
    }
    weights, drawdown_control = apply_drawdown_control(
        weights,
        backtest.equity_curve,
        config=drawdown_config,
        return_details=True,
    )
    risk = portfolio_risk_report(
        weights,
        returns=returns,
        factor_exposures=factor_exposures,
        config={
            "lookback": 252,
        },
    )
    risk["simulated_strategy_drawdown"] = float(backtest.drawdown.iloc[-1])
    risk["simulated_drawdown_risk_multiplier"] = drawdown_control["risk_multiplier"]
    risk["pre_control_gross_exposure"] = float(uncontrolled_weights.abs().sum())
    risk["post_control_gross_exposure"] = float(weights.abs().sum())
    risk["tradability_exclusions"] = dict(
        uncontrolled_weights.attrs.get("tradability_exclusions", {})
    )
    scenarios = run_scenario_analysis(weights, factor_exposures=factor_exposures)
    stress = historical_stress_test(weights, returns)
    benchmark_config = {**backtest_config, "evaluation_start": start_date}
    benchmarks = compare_benchmarks(backtest, returns, config=benchmark_config)
    research_questions = _research_question_status(lead_lag, transitions, benchmarks)
    catalysts = build_catalyst_calendar(config.catalysts, as_of=as_of_date, horizon_days=45)
    health = model_health_report(
        factors,
        regimes.scores,
        strategy_returns=backtest.returns,
        turnover=backtest.turnover,
        data_quality_score=quality.score,
    )

    raw_dir = destination / "data" / "raw"
    processed_dir = destination / "data" / "processed"
    diagnostics_dir = destination / "diagnostics"
    raw_snapshot_path = raw_dir / "observations.csv.gz"
    raw_snapshot_manifest = save_snapshot(
        observations,
        raw_snapshot_path,
        metadata={
            "mode": selected_mode,
            "origin_mode": observation_attrs.get(
                "origin_mode", observation_attrs.get("mode", selected_mode)
            ),
            "synthetic": bool(observation_attrs.get("synthetic", False)),
            "ingestion_issues": observation_attrs.get("ingestion_issues", []),
            "timestamp_quality": observation_attrs.get("timestamp_quality"),
            "as_of": as_of_date.isoformat(),
        },
    )
    raw_snapshot_manifest_path = raw_snapshot_path.with_name(
        f"{raw_snapshot_path.name}.manifest.json"
    )
    _write_csv(market, processed_dir / "market_point_in_time.csv.gz")
    _write_csv(aligned.availability, processed_dir / "availability_timestamps.csv.gz")
    _write_csv(aligned.age_days, processed_dir / "data_age_days.csv.gz")
    _write_csv(aligned.metadata, diagnostics_dir / "series_metadata.csv")
    _write_csv(
        invalid_value_mask.astype(int),
        diagnostics_dir / "quarantined_value_mask.csv.gz",
    )
    _write_csv(features, processed_dir / "features.csv.gz")
    _write_csv(factors, processed_dir / "mvp_factors.csv")
    _write_csv(economic_features, processed_dir / "economic_features.csv.gz")
    _write_csv(regimes.scores, processed_dir / "regime_scores.csv")
    _write_csv(regimes.raw_scores, diagnostics_dir / "regime_raw_scores.csv")
    _write_csv(regimes.coverage, diagnostics_dir / "regime_coverage.csv")
    _write_csv(regimes.contributions, diagnostics_dir / "regime_contributions.csv.gz", index=False)
    _write_csv(factor_coverage(factors), diagnostics_dir / "factor_coverage.csv")
    if statistical is not None:
        _write_csv(statistical.probabilities, diagnostics_dir / "statistical_regime_probabilities.csv.gz")
        _write_csv(statistical.coefficients, diagnostics_dir / "statistical_regime_coefficients.csv")
        _write_json(statistical.metadata, diagnostics_dir / "statistical_regime_metadata.json")
    for horizon, matrix in transitions.transition_matrices.items():
        _write_csv(matrix, diagnostics_dir / f"transition_matrix_{horizon}.csv")
    for horizon, history in transitions.forecast_history.items():
        _write_csv(history, diagnostics_dir / f"transition_forecast_{horizon}.csv.gz")
    _write_json(transitions.metadata, diagnostics_dir / "transition_metadata.json")
    _write_csv(signals, processed_dir / "signals.csv.gz", index=False)
    _write_csv(weights, processed_dir / "current_portfolio.csv")
    _write_csv(
        uncontrolled_weights,
        diagnostics_dir / "uncontrolled_current_portfolio.csv",
    )
    _write_csv(backtest.equity_curve, processed_dir / "backtest_equity.csv")
    _write_csv(backtest.returns, processed_dir / "backtest_returns.csv")
    _write_csv(backtest.positions, processed_dir / "backtest_positions.csv.gz")
    _write_csv(backtest.costs, processed_dir / "backtest_costs.csv")
    _write_json(backtest.metrics, processed_dir / "backtest_metrics.json")
    _write_csv(benchmarks, diagnostics_dir / "benchmark_comparison.csv")
    _write_csv(scenarios, diagnostics_dir / "scenario_analysis.csv")
    _write_csv(stress, diagnostics_dir / "historical_stress_tests.csv")
    _write_csv(catalysts, processed_dir / "upcoming_catalysts.csv", index=False)
    _write_json(quality.as_dict(), destination / "data_quality.json")
    _write_json(health, diagnostics_dir / "model_health.json")
    _write_csv(
        lead_lag.cross_correlations,
        diagnostics_dir / "lead_lag_cross_correlations.csv",
        index=False,
    )
    _write_csv(
        lead_lag.distributed_lag_coefficients,
        diagnostics_dir / "lead_lag_distributed_regressions.csv",
        index=False,
    )
    _write_csv(
        lead_lag.granger_tests,
        diagnostics_dir / "lead_lag_granger_tests.csv",
        index=False,
    )
    _write_csv(
        lead_lag.metadata,
        diagnostics_dir / "lead_lag_metadata.csv",
        index=False,
    )
    _write_csv(
        research_questions,
        diagnostics_dir / "research_question_status.csv",
        index=False,
    )
    drift = health.get("factor_drift")
    if isinstance(drift, pd.DataFrame):
        _write_csv(drift, diagnostics_dir / "factor_drift.csv", index=False)
    catalyst_quality_values = (
        sorted(catalysts["calendar_quality"].dropna().astype(str).unique().tolist())
        if "calendar_quality" in catalysts
        else []
    )
    upstream_manifest = observation_attrs.get("snapshot_manifest", {})
    upstream_snapshot_sha = (
        upstream_manifest.get("sha256")
        if isinstance(upstream_manifest, Mapping)
        else None
    )
    effective_runtime = {
        "mode": selected_mode,
        "as_of": as_of_date,
        "evaluation_start": start_date,
        "fetch_start": fetch_start,
        "warmup_business_days": warmup_business_days,
        "decision_cutoff": as_of_date
        + pd.to_timedelta(config.runtime.decision_time_utc),
        "upstream_snapshot_sha256": upstream_snapshot_sha,
        "run_statistical": bool(run_statistical),
    }
    invocation = {
        "cache_path": observation_attrs.get("snapshot_path", cache_path),
        "output_dir": destination,
        "overwrite": bool(overwrite),
        "fail_on_quality": bool(fail_on_quality),
        "record_research_log": bool(record_research_log),
    }
    configuration_snapshot = _configuration_snapshot(config, effective_runtime)
    configuration_snapshot["invocation"] = invocation
    _write_json(configuration_snapshot, destination / "configuration_snapshot.json")

    run_metadata = {
        **effective_runtime,
        "invocation": invocation,
        "latest_market_date": market.index.max(),
        "random_seed": config.runtime.random_seed,
        "point_in_time_safe": True,
        "execution_lag": backtest_config.get("execution_lag_days", backtest_config.get("execution_lag", 1)),
        "quality_confidence_multiplier": quality_multiplier,
        "trade_min_signal": float(portfolio_config.get("min_signal", 0.05)),
        "trade_confidence_floor": float(
            portfolio_config.get("confidence_floor", 0.0)
        ),
        "origin_mode": observation_attrs.get(
            "origin_mode", observation_attrs.get("mode", selected_mode)
        ),
        "synthetic": bool(observation_attrs.get("synthetic", False)),
        "statistical_regime_role": "diagnostic surrogate-label model; transparent scores remain the decision baseline",
        "catalyst_calendar_quality": (
            ",".join(catalyst_quality_values)
            if catalyst_quality_values
            else "unavailable"
        ),
    }
    reports = generate_reports(
        destination,
        as_of=as_of_date,
        system_version=__version__,
        market=market,
        factors=signal_factors,
        regime_scores=regimes.scores,
        contributions=regimes,
        transition_probabilities=transitions,
        signals=signals,
        portfolio_weights=weights,
        risk_report=risk,
        backtest=backtest,
        scenarios=scenarios.reset_index(),
        data_quality=quality.as_dict(),
        source_label=_source_label(selected_mode, observation_attrs),
        run_metadata=run_metadata,
        catalysts=catalysts,
        benchmarks=benchmarks,
        stress_tests=stress,
        model_health=health,
        lead_lag=lead_lag,
        research_questions=research_questions,
        series_metadata=aligned.metadata,
    )

    output_listing = {
        "report_html": str(Path(reports["html"]).relative_to(destination)),
        "report_markdown": str(Path(reports["markdown"]).relative_to(destination)),
        "technical_report_markdown": str(
            Path(reports["technical_markdown"]).relative_to(destination)
        ),
        "monitor_json": str(Path(reports["json"]).relative_to(destination)),
        "chart_count": len(reports["charts"]),
    }
    upstream_snapshot_manifest = observation_attrs.get("snapshot_manifest")
    if not isinstance(upstream_snapshot_manifest, Mapping):
        upstream_snapshot_manifest = None
    upstream_snapshot_manifest_sha256 = None
    upstream_snapshot_path = observation_attrs.get("snapshot_path")
    if upstream_snapshot_path:
        source_snapshot = Path(str(upstream_snapshot_path)).expanduser().resolve()
        source_manifest = source_snapshot.with_name(f"{source_snapshot.name}.manifest.json")
        if source_manifest.is_file():
            upstream_snapshot_manifest_sha256 = file_hash(source_manifest)
    if upstream_snapshot_manifest and upstream_snapshot_manifest_sha256 is None:
        # A connector may carry a parsed manifest without exposing a local
        # file.  Preserve a deterministic content link in that case.
        upstream_snapshot_manifest_sha256 = stable_json_hash(upstream_snapshot_manifest)
    manifest = build_manifest(
        project_root=config.project_root,
        configuration=configuration_snapshot,
        data=market,
        as_of=as_of_date,
        mode=selected_mode,
        seed=config.runtime.random_seed,
        outputs=output_listing,
        raw_observations=observations,
        availability=aligned.availability,
        raw_snapshot_manifest=raw_snapshot_manifest,
        raw_snapshot_manifest_sha256=file_hash(raw_snapshot_manifest_path),
        raw_snapshot_manifest_path=str(raw_snapshot_manifest_path.relative_to(destination)),
        upstream_snapshot_manifest=upstream_snapshot_manifest,
        upstream_snapshot_manifest_sha256=upstream_snapshot_manifest_sha256,
        reproducibility_state=reproducibility_state,
    )
    manifest["quality"] = {"score": quality.score, "status": quality.status}
    manifest["model_health"] = {
        "status": health["status"],
        "warnings": health["warnings"],
    }
    write_manifest(destination / "manifest.json", manifest)

    if record_research_log:
        log_path = config.paths.reports_dir / "research_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_record = {
            "run_id": destination.name,
            "output_dir": str(destination),
            "generated_at_utc": manifest["generated_at_utc"],
            "as_of": manifest["as_of"],
            "mode": selected_mode,
            "system_version": __version__,
            "code_revision": manifest["code_revision"],
            "configuration_sha256": manifest["configuration_sha256"],
            "data_snapshot_sha256": manifest["data_snapshot_sha256"],
            "dominant_regime": str(regimes.scores.iloc[-1].idxmax()),
            "signals_file": str(processed_dir / "signals.csv.gz"),
            "portfolio_file": str(processed_dir / "current_portfolio.csv"),
            "realized_outcome_status": "pending",
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record, sort_keys=True, allow_nan=False) + "\n")

    return PipelineResult(
        output_dir=destination,
        observations=observations,
        market=market,
        features=features,
        factors=factors,
        signal_factors=signal_factors,
        regimes=regimes,
        statistical_regimes=statistical,
        transitions=transitions,
        signals=signals,
        portfolio_weights=weights,
        backtest=backtest,
        risk=risk,
        scenarios=scenarios,
        stress_tests=stress,
        benchmarks=benchmarks,
        catalysts=catalysts,
        model_health=health,
        lead_lag=lead_lag,
        research_questions=research_questions,
        quality=quality,
        manifest=manifest,
        reports=reports,
    )


__all__ = ["PipelineResult", "run_pipeline"]
