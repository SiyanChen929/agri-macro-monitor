"""Deterministic data-quality checks and confidence haircuts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..data_models import CANONICAL_OBSERVATION_COLUMNS, SeriesSpec, normalize_observation_frame


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str
    series_id: str | None = None
    date: pd.Timestamp | None = None
    count: int = 1
    penalty: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "series_id": self.series_id,
            "date": self.date.isoformat() if self.date is not None else None,
            "count": int(self.count),
            "penalty": self.penalty,
        }


_DEFAULT_PENALTIES = {
    Severity.INFO: 0.0,
    Severity.WARNING: 3.0,
    Severity.ERROR: 12.0,
    Severity.CRITICAL: 30.0,
}


@dataclass
class DataQualityReport:
    """Portable quality report used to haircut model/trade confidence."""

    score: float
    status: str
    issues: list[QualityIssue] = field(default_factory=list)
    series: pd.DataFrame = field(default_factory=pd.DataFrame)
    checked_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC").tz_localize(None))

    @property
    def confidence_multiplier(self) -> float:
        """Conservative 0..1 multiplier for downstream signal confidence."""

        if self.status == "failed":
            return 0.0
        return float(np.clip(self.score / 100.0, 0.0, 1.0))

    @property
    def passed(self) -> bool:
        return self.status != "failed"

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def errors(self) -> list[QualityIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity in {Severity.ERROR, Severity.CRITICAL}
        ]

    def as_dict(self) -> dict[str, Any]:
        def json_scalar(value: Any) -> Any:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                return pd.Timestamp(value).isoformat()
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                return None
            return value

        raw_records = self.series.reset_index().to_dict(orient="records") if not self.series.empty else []
        records = [
            {key: json_scalar(value) for key, value in record.items()}
            for record in raw_records
        ]
        return {
            "score": round(float(self.score), 4),
            "status": self.status,
            "confidence_multiplier": round(self.confidence_multiplier, 6),
            "checked_at": self.checked_at.isoformat(),
            "issues": [issue.as_dict() for issue in self.issues],
            "series": records,
        }

    @classmethod
    def from_issues(
        cls,
        issues: Iterable[QualityIssue],
        *,
        series: pd.DataFrame | None = None,
        warning_score: float = 85.0,
        failure_score: float = 60.0,
        penalties: Mapping[Severity | str, float] | None = None,
    ) -> "DataQualityReport":
        issue_list = list(issues)
        penalty_table = dict(_DEFAULT_PENALTIES)
        for severity, value in dict(penalties or {}).items():
            penalty_table[
                severity if isinstance(severity, Severity) else Severity(str(severity))
            ] = float(value)
        penalty = sum(
            issue.penalty if issue.penalty is not None else penalty_table[issue.severity]
            for issue in issue_list
        )
        score = float(np.clip(100.0 - penalty, 0.0, 100.0))
        has_error = any(
            issue.severity in {Severity.ERROR, Severity.CRITICAL}
            for issue in issue_list
        )
        has_warning = any(issue.severity == Severity.WARNING for issue in issue_list)
        status = (
            "failed"
            if has_error or score < failure_score
            else "warning"
            if has_warning or score < warning_score
            else "healthy"
        )
        return cls(score=score, status=status, issues=issue_list, series=series if series is not None else pd.DataFrame())

    @classmethod
    def merge(
        cls,
        *reports: "DataQualityReport",
        warning_score: float = 85.0,
        failure_score: float = 60.0,
        penalties: Mapping[Severity | str, float] | None = None,
    ) -> "DataQualityReport":
        """Combine reports without double-counting their already-reduced scores."""

        if not reports:
            return cls(score=100.0, status="healthy")
        issues = [issue for report in reports for issue in report.issues]
        series_frames = [report.series for report in reports if not report.series.empty]
        series = pd.concat(series_frames).groupby(level=0).last() if series_frames else pd.DataFrame()
        result = cls.from_issues(
            issues,
            series=series,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )
        result.score = min(result.score, *(report.score for report in reports))
        if any(report.status == "failed" for report in reports):
            result.status = "failed"
        elif result.status == "healthy" and any(report.status == "warning" for report in reports):
            result.status = "warning"
        return result


def _spec_mapping(specs: Mapping[str, SeriesSpec] | Iterable[SeriesSpec] | None) -> dict[str, SeriesSpec]:
    if specs is None:
        return {}
    if isinstance(specs, Mapping):
        return dict(specs)
    return {spec.series_id: spec for spec in specs}


def validate_observations(
    frame: pd.DataFrame,
    specs: Mapping[str, SeriesSpec] | Iterable[SeriesSpec] | None = None,
    *,
    as_of: Any | None = None,
    abnormal_jump_z: float = 8.0,
    min_critical_coverage: float = 0.80,
    warning_score: float = 85.0,
    failure_score: float = 60.0,
    penalties: Mapping[Severity | str, float] | None = None,
    stale_penalty: float = 4.0,
) -> DataQualityReport:
    """Validate canonical raw vintages before any signal is calculated.

    Checks are deliberately non-mutating.  Invalid data are reported rather
    than silently repaired; alignment subsequently masks stale observations.
    """

    spec_map = _spec_mapping(specs)
    issues: list[QualityIssue] = []
    missing_columns = sorted(set(CANONICAL_OBSERVATION_COLUMNS[:6]).difference(frame.columns))
    if missing_columns:
        issues.append(
            QualityIssue(
                code="missing_required_columns",
                severity=Severity.CRITICAL,
                message=f"Missing required columns: {missing_columns}",
                count=len(missing_columns),
            )
        )
        return DataQualityReport.from_issues(
            issues,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )

    try:
        data = normalize_observation_frame(frame)
    except (TypeError, ValueError) as exc:
        issues.append(QualityIssue("schema_error", Severity.CRITICAL, str(exc)))
        return DataQualityReport.from_issues(
            issues,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )

    bad_dates = data[["observation_date", "publication_date", "effective_date", "revision_date"]].isna().any(axis=1)
    if bad_dates.any():
        issues.append(
            QualityIssue(
                "invalid_timestamp",
                Severity.ERROR,
                "Rows contain missing or unparseable timestamps",
                count=int(bad_dates.sum()),
            )
        )
    missing_values = data["value"].isna() | ~np.isfinite(data["value"].astype(float))
    if missing_values.any():
        issues.append(
            QualityIssue(
                "non_finite_value",
                Severity.ERROR,
                "Rows contain missing or non-finite numeric values",
                count=int(missing_values.sum()),
            )
        )

    identity = ["series_id", "observation_date", "publication_date", "revision_date"]
    duplicate = data.duplicated(identity, keep=False)
    if duplicate.any():
        issues.append(
            QualityIssue(
                "duplicate_vintage",
                Severity.ERROR,
                "Duplicate series/observation/publication/revision identities detected",
                count=int(duplicate.sum()),
            )
        )

    revision_before_publication = data["revision_date"] < data["publication_date"]
    if revision_before_publication.any():
        issues.append(
            QualityIssue(
                "revision_before_publication",
                Severity.CRITICAL,
                "A revision cannot be available before the initial publication",
                count=int(revision_before_publication.sum()),
            )
        )

    unknown = sorted(set(data["series_id"].dropna().astype(str)).difference(spec_map)) if spec_map else []
    if unknown:
        issues.append(
            QualityIssue(
                "unconfigured_series",
                Severity.WARNING,
                f"Unconfigured series will use conservative defaults: {unknown}",
                count=len(unknown),
            )
        )

    ingestion_issues = frame.attrs.get("ingestion_issues", [])
    for payload in ingestion_issues:
        if isinstance(payload, QualityIssue):
            issues.append(payload)
        else:
            issues.append(
                QualityIssue(
                    code=str(payload.get("code", "ingestion_error")),
                    severity=Severity(str(payload.get("severity", "error"))),
                    message=str(payload.get("message", "An ingestion adapter failed")),
                    series_id=payload.get("series_id"),
                )
            )

    series_rows: list[dict[str, Any]] = []
    current_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    for series_id, subset in data.groupby("series_id", sort=True):
        name = str(series_id)
        spec = spec_map.get(name)
        clean = subset.loc[~missing_values.loc[subset.index]].sort_values(["observation_date", "revision_date"])
        latest_vintages = clean.drop_duplicates("observation_date", keep="last")
        first_date = latest_vintages["observation_date"].min() if not latest_vintages.empty else pd.NaT
        last_obs = latest_vintages["observation_date"].max() if not latest_vintages.empty else pd.NaT
        last_release = (
            clean[["publication_date", "revision_date", "effective_date"]].max(axis=1).max()
            if not clean.empty
            else pd.NaT
        )
        age = int(max((current_date - last_release.normalize()).days, 0)) if pd.notna(last_release) else np.nan

        below = above = pd.Series(False, index=clean.index)
        if spec and spec.min_value is not None:
            below = clean["value"] < spec.min_value
        if spec and spec.max_value is not None:
            above = clean["value"] > spec.max_value
        impossible = below | above
        if impossible.any():
            issues.append(
                QualityIssue(
                    "impossible_value",
                    Severity.ERROR if spec and spec.critical else Severity.WARNING,
                    "Values fall outside configured economic bounds",
                    series_id=name,
                    count=int(impossible.sum()),
                )
            )

        if spec is not None:
            reported_frequencies = set(clean["frequency"].dropna().astype(str).str.lower())
            if reported_frequencies and reported_frequencies != {spec.frequency.value}:
                issues.append(
                    QualityIssue(
                        "frequency_mismatch",
                        Severity.WARNING,
                        f"Reported frequencies {sorted(reported_frequencies)} differ from configured {spec.frequency.value}",
                        series_id=name,
                        penalty=1.0,
                    )
                )
            if spec.max_abs_daily_return is not None and len(latest_vintages) >= 2:
                ordered_values = latest_vintages.set_index("observation_date")["value"].sort_index()
                returns = ordered_values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                breaches = returns.abs() > spec.max_abs_daily_return
                if breaches.any():
                    issues.append(
                        QualityIssue(
                            "return_limit_breach",
                            Severity.WARNING,
                            f"Observed returns exceed configured absolute limit {spec.max_abs_daily_return:.1%}",
                            series_id=name,
                            count=int(breaches.sum()),
                            penalty=min(3.0, 0.5 * int(breaches.sum())),
                        )
                    )

        if "timestamp_quality" in clean:
            approximate_final = clean["timestamp_quality"].astype(str).eq(
                "approximated_configured_lag_final_value"
            )
            if approximate_final.any():
                issues.append(
                    QualityIssue(
                        "non_vintage_history",
                        Severity.WARNING,
                        "Final revised values use an approximated publication lag; backtest confidence is reduced",
                        series_id=name,
                        count=int(approximate_final.sum()),
                        penalty=5.0,
                    )
                )
        if "roll_method" in clean and clean["roll_method"].astype(str).eq("vendor_continuous_proxy").any():
            issues.append(
                QualityIssue(
                    "unvalidated_futures_roll",
                    Severity.WARNING,
                    "Vendor continuous futures proxy has no independently validated roll-cost history",
                    series_id=name,
                    penalty=1.0,
                )
            )

        abnormal_count = 0
        if len(latest_vintages) >= 10:
            changes = latest_vintages.set_index("observation_date")["value"].diff()
            median = changes.median()
            mad = (changes - median).abs().median()
            if pd.notna(mad) and mad > 0:
                robust_z = 0.67448975 * (changes - median) / mad
                abnormal = robust_z.abs() > abnormal_jump_z
                abnormal_count = int(abnormal.sum())
                if abnormal_count:
                    issues.append(
                        QualityIssue(
                            "abnormal_jump",
                            Severity.WARNING,
                            f"{abnormal_count} changes exceed robust z={abnormal_jump_z:g}",
                            series_id=name,
                            count=abnormal_count,
                            penalty=min(3.0, 0.5 * abnormal_count),
                        )
                    )

        stale = bool(spec and pd.notna(age) and age > spec.staleness_limit)
        if stale:
            issues.append(
                QualityIssue(
                    "stale_latest_release",
                    Severity.ERROR if spec.critical else Severity.WARNING,
                    f"Latest release is {age} days old; limit is {spec.staleness_limit}",
                    series_id=name,
                    date=last_release,
                    penalty=float(stale_penalty) * (2.0 if spec.critical else 0.5),
                )
            )
        series_rows.append(
            {
                "series_id": name,
                "rows": int(len(subset)),
                "observations": int(latest_vintages["observation_date"].nunique()),
                "vintages": int(len(clean)),
                "first_observation": first_date,
                "last_observation": last_obs,
                "last_available": last_release,
                "data_age_days": age,
                "stale": stale,
                "abnormal_jumps": abnormal_count,
                "critical": bool(spec.critical) if spec else False,
            }
        )

    present = set(data["series_id"].dropna().astype(str))
    missing_critical = [name for name, spec in spec_map.items() if spec.critical and name not in present]
    for name in missing_critical:
        issues.append(
            QualityIssue(
                "missing_critical_series",
                Severity.CRITICAL,
                "Configured critical series has no observations",
                series_id=name,
            )
        )
        series_rows.append({"series_id": name, "rows": 0, "critical": True, "stale": True})

    critical_count = sum(spec.critical for spec in spec_map.values())
    critical_present = critical_count - len(missing_critical)
    critical_coverage = critical_present / critical_count if critical_count else 1.0
    if critical_coverage < min_critical_coverage:
        issues.append(
            QualityIssue(
                "insufficient_critical_coverage",
                Severity.CRITICAL,
                f"Critical-series coverage {critical_coverage:.1%} is below {min_critical_coverage:.1%}",
            )
        )

    series_frame = pd.DataFrame(series_rows).set_index("series_id") if series_rows else pd.DataFrame()
    return DataQualityReport.from_issues(
        issues,
        series=series_frame,
        warning_score=warning_score,
        failure_score=failure_score,
        penalties=penalties,
    )


def validate_aligned_data(
    data: pd.DataFrame,
    availability: pd.DataFrame,
    age_days: pd.DataFrame,
    specs: Mapping[str, SeriesSpec] | None = None,
    *,
    decision_at: Iterable[Any] | pd.DatetimeIndex | None = None,
    min_critical_coverage: float = 0.50,
    warning_score: float = 85.0,
    failure_score: float = 60.0,
    penalties: Mapping[Severity | str, float] | None = None,
) -> DataQualityReport:
    """Audit an aligned panel for lookahead, coverage and stale leakage."""

    issues: list[QualityIssue] = []
    spec_map = dict(specs or {})
    if not data.index.equals(availability.index) or not data.columns.equals(availability.columns):
        issues.append(QualityIssue("alignment_shape_mismatch", Severity.CRITICAL, "Value and availability panels differ"))
        return DataQualityReport.from_issues(
            issues,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )
    if not data.index.equals(age_days.index) or not data.columns.equals(age_days.columns):
        issues.append(QualityIssue("age_shape_mismatch", Severity.CRITICAL, "Value and age panels differ"))
        return DataQualityReport.from_issues(
            issues,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )
    if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing:
        issues.append(QualityIssue("invalid_calendar", Severity.CRITICAL, "Aligned index must be increasing DatetimeIndex"))
        return DataQualityReport.from_issues(
            issues,
            warning_score=warning_score,
            failure_score=failure_score,
            penalties=penalties,
        )

    if decision_at is None:
        decision_index = data.index + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
        decision_index = pd.DatetimeIndex(pd.to_datetime(list(decision_at), errors="coerce", utc=True)).tz_localize(None)
        if len(decision_index) != len(data.index):
            issues.append(
                QualityIssue(
                    "decision_calendar_shape_mismatch",
                    Severity.CRITICAL,
                    "Decision timestamps and aligned values have different row counts",
                )
            )
            return DataQualityReport.from_issues(
                issues,
                warning_score=warning_score,
                failure_score=failure_score,
                penalties=penalties,
            )
    decision_matrix = np.broadcast_to(decision_index.values[:, None], data.shape)
    availability_values = availability.to_numpy(dtype="datetime64[ns]")
    future = (~pd.isna(availability_values)) & (availability_values > decision_matrix)
    if future.any():
        issues.append(
            QualityIssue(
                "lookahead_detected",
                Severity.CRITICAL,
                "Aligned values include vintages not available on the decision date",
                count=int(future.sum()),
            )
        )

    rows: list[dict[str, Any]] = []
    for column in data.columns:
        coverage = float(data[column].notna().mean()) if len(data) else 0.0
        spec = spec_map.get(str(column))
        if spec and spec.critical and coverage < float(min_critical_coverage):
            issues.append(
                QualityIssue(
                    "low_aligned_coverage",
                    Severity.ERROR,
                    f"Aligned coverage is only {coverage:.1%}",
                    series_id=str(column),
                    penalty=8.0,
                )
            )
        limit = spec.staleness_limit if spec else None
        stale_leak = (age_days[column] > limit) & data[column].notna() if limit is not None else pd.Series(False, index=data.index)
        if stale_leak.any():
            issues.append(
                QualityIssue(
                    "stale_value_leak",
                    Severity.CRITICAL,
                    "Values remained populated after configured staleness limit",
                    series_id=str(column),
                    count=int(stale_leak.sum()),
                )
            )
        rows.append(
            {
                "series_id": str(column),
                "aligned_coverage": coverage,
                "first_valid": data[column].first_valid_index(),
                "last_valid": data[column].last_valid_index(),
                "max_age_days": age_days[column].max(skipna=True),
                "critical": bool(spec.critical) if spec else False,
            }
        )
    series_frame = pd.DataFrame(rows).set_index("series_id") if rows else pd.DataFrame()
    return DataQualityReport.from_issues(
        issues,
        series=series_frame,
        warning_score=warning_score,
        failure_score=failure_score,
        penalties=penalties,
    )
