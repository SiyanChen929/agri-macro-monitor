"""Canonical data contracts for the agricultural macro research system.

The ingestion boundary is deliberately long-form and revision aware.  Downstream
research code should consume :class:`AlignmentResult.data`, a daily wide frame
that has already been filtered to information available at each decision date.

All timestamps are represented as timezone-naive UTC-equivalent pandas
``Timestamp`` objects.  A row becomes usable only at the maximum of its
``publication_date``, ``revision_date`` and ``effective_date``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imported only for static type checkers
    from .validation.data_quality import DataQualityReport


class SeriesFrequency(str, Enum):
    """Supported release frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    IRREGULAR = "irregular"

    @property
    def default_max_staleness_days(self) -> int:
        return {
            self.DAILY: 4,
            self.WEEKLY: 10,
            self.MONTHLY: 45,
            self.QUARTERLY: 120,
            self.ANNUAL: 400,
            self.IRREGULAR: 30,
        }[self]


CANONICAL_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "series_id",
    "observation_date",
    "publication_date",
    "effective_date",
    "revision_date",
    "value",
    "source",
    "frequency",
    "unit",
    "vintage_id",
)

CANONICAL_WIDE_SERIES: tuple[str, ...] = (
    "wheat",
    "corn",
    "soybeans",
    "natural_gas",
    "urea",
    "cf",
    "ntr",
    "mos",
    "dxy",
    "us_2y",
    "breakeven_5y",
    "crop_condition",
    "yield_revision",
    "exporter_stocks_to_use",
    "policy_stress",
    "weather_stress",
)

ObservationFrame: TypeAlias = pd.DataFrame


def _utc_naive(value: Any) -> pd.Timestamp:
    """Coerce one timestamp to UTC and remove the timezone marker."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


@dataclass(frozen=True)
class Observation:
    """One published vintage of one economic or market observation."""

    series_id: str
    observation_date: pd.Timestamp
    publication_date: pd.Timestamp
    effective_date: pd.Timestamp
    revision_date: pd.Timestamp
    value: float
    source: str
    frequency: SeriesFrequency
    unit: str = "index"
    vintage_id: str = "initial"

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, SeriesFrequency):
            object.__setattr__(self, "frequency", SeriesFrequency(str(self.frequency).lower()))

    @property
    def available_at(self) -> pd.Timestamp:
        """First instant at which this particular vintage may be used."""

        return max(
            _utc_naive(self.publication_date),
            _utc_naive(self.effective_date),
            _utc_naive(self.revision_date),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "observation_date": _utc_naive(self.observation_date),
            "publication_date": _utc_naive(self.publication_date),
            "effective_date": _utc_naive(self.effective_date),
            "revision_date": _utc_naive(self.revision_date),
            "value": float(self.value),
            "source": self.source,
            "frequency": self.frequency.value,
            "unit": self.unit,
            "vintage_id": self.vintage_id,
        }


@dataclass(frozen=True)
class SeriesSpec:
    """Configuration and validation rules for one canonical series."""

    series_id: str
    frequency: SeriesFrequency
    source: str = "demo"
    source_id: str | None = None
    unit: str = "index"
    publication_lag_days: int = 0
    max_staleness_days: int | None = None
    critical: bool = False
    min_value: float | None = None
    max_value: float | None = None
    max_abs_daily_return: float | None = None
    price_scale: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, SeriesFrequency):
            object.__setattr__(self, "frequency", SeriesFrequency(str(self.frequency).lower()))
        if self.publication_lag_days < 0:
            raise ValueError("publication_lag_days cannot be negative")
        if self.max_staleness_days is not None and self.max_staleness_days < 0:
            raise ValueError("max_staleness_days cannot be negative")
        if self.price_scale <= 0:
            raise ValueError("price_scale must be positive")
        if not math.isfinite(float(self.price_scale)):
            raise ValueError("price_scale must be finite")
        if self.min_value is not None and not math.isfinite(float(self.min_value)):
            raise ValueError("min_value must be finite")
        if self.max_value is not None and not math.isfinite(float(self.max_value)):
            raise ValueError("max_value must be finite")
        if (
            self.min_value is not None
            and self.max_value is not None
            and float(self.min_value) > float(self.max_value)
        ):
            raise ValueError("min_value cannot exceed max_value")
        if self.max_abs_daily_return is not None:
            limit = float(self.max_abs_daily_return)
            if not math.isfinite(limit) or limit < 0.0:
                raise ValueError("max_abs_daily_return must be finite and non-negative")

    @property
    def staleness_limit(self) -> int:
        if self.max_staleness_days is not None:
            return int(self.max_staleness_days)
        return self.frequency.default_max_staleness_days

    @classmethod
    def from_mapping(cls, series_id: str, values: Mapping[str, Any]) -> "SeriesSpec":
        known = {
            "frequency",
            "source",
            "source_id",
            "unit",
            "publication_lag_days",
            "max_staleness_days",
            "critical",
            "min_value",
            "max_value",
            "max_abs_daily_return",
            "price_scale",
            "description",
        }
        # Adapter-location fields are consumed by the ingestion dispatcher,
        # not by SeriesSpec itself, but are valid in the same per-series block.
        adapter_fields = {"location", "path", "date_column", "value_column"}
        unknown = sorted(set(map(str, values)).difference(known | adapter_fields))
        if unknown:
            raise ValueError(
                f"series {series_id!r} contains unknown configuration key(s): "
                f"{unknown}. Unknown keys are rejected so publication/staleness "
                "typos cannot silently use defaults."
            )
        kwargs = {key: value for key, value in values.items() if key in known}
        kwargs["frequency"] = SeriesFrequency(str(kwargs.get("frequency", "daily")).lower())
        return cls(series_id=series_id, **kwargs)


@dataclass(frozen=True)
class PolicyEvent:
    """Structured government intervention event."""

    date: pd.Timestamp
    country: str
    commodity: str
    policy_type: str
    direction: float
    magnitude: float
    duration_days: int
    confidence: float
    source: str
    expected_domestic_effect: str
    expected_global_effect: str
    trade_share: float = 1.0
    importance: float = 1.0

    def stress_contribution(self, age_days: int = 0, half_life_days: float = 30.0) -> float:
        """Signed, exponentially decayed contribution to policy stress."""

        import math

        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        decay = math.exp(-math.log(2.0) * max(age_days, 0) / half_life_days)
        return (
            float(self.direction)
            * float(self.magnitude)
            * float(self.confidence)
            * float(self.trade_share)
            * float(self.importance)
            * decay
        )


@dataclass
class AlignmentResult:
    """Point-in-time daily panel and its audit information.

    ``data`` contains values only and is the canonical input to feature code.
    The three companion frames have exactly the same index/columns:

    * ``availability``: timestamp of the selected vintage;
    * ``observation_dates``: economic observation period selected;
    * ``age_days``: calendar days since that vintage became available.
    """

    data: pd.DataFrame
    quality: "DataQualityReport"
    metadata: pd.DataFrame
    availability: pd.DataFrame = field(default_factory=pd.DataFrame)
    observation_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    age_days: pd.DataFrame = field(default_factory=pd.DataFrame)

    def with_age_features(self, suffix: str = "__age_days") -> pd.DataFrame:
        """Return values plus explicit age features without mutating ``data``."""

        ages = self.age_days.rename(columns=lambda name: f"{name}{suffix}")
        return self.data.join(ages, how="left")

    def as_of(self, date: Any) -> pd.Series:
        """Return the latest aligned row no later than ``date``."""

        timestamp = _utc_naive(date).normalize()
        eligible = self.data.index[self.data.index <= timestamp]
        if len(eligible) == 0:
            raise KeyError(f"No aligned data are available on or before {timestamp.date()}")
        return self.data.loc[eligible[-1]]


def empty_observation_frame() -> ObservationFrame:
    """Create an empty canonical frame with stable column ordering."""

    return pd.DataFrame(columns=list(CANONICAL_OBSERVATION_COLUMNS))


def observations_to_frame(observations: list[Observation]) -> ObservationFrame:
    """Convert typed observations into the canonical tabular representation."""

    if not observations:
        return empty_observation_frame()
    return pd.DataFrame.from_records(
        [item.as_dict() for item in observations],
        columns=list(CANONICAL_OBSERVATION_COLUMNS),
    )


def normalize_observation_frame(frame: pd.DataFrame, *, copy: bool = True) -> ObservationFrame:
    """Normalize column order and scalar dtypes without applying PIT alignment.

    Missing ``effective_date`` and ``revision_date`` values are conservatively
    replaced with ``observation_date`` and ``publication_date`` respectively.
    Missing required identity/value fields remain errors.
    """

    required = {"series_id", "observation_date", "publication_date", "value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Observation frame is missing required columns: {missing}")

    result = frame.copy() if copy else frame
    if "effective_date" not in result:
        result["effective_date"] = result["observation_date"]
    if "revision_date" not in result:
        result["revision_date"] = result["publication_date"]
    defaults: dict[str, Any] = {
        "source": "unknown",
        "frequency": SeriesFrequency.IRREGULAR.value,
        "unit": "index",
        "vintage_id": "initial",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default

    for column in ("observation_date", "publication_date", "effective_date", "revision_date"):
        parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
        result[column] = parsed.dt.tz_localize(None)
    result["series_id"] = result["series_id"].astype("string")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    for column in ("source", "frequency", "unit", "vintage_id"):
        result[column] = result[column].astype("string")

    ordered = list(CANONICAL_OBSERVATION_COLUMNS)
    extras = [column for column in result.columns if column not in ordered]
    return result.loc[:, ordered + extras]
