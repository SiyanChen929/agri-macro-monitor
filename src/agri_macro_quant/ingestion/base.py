"""Adapter interface shared by offline and public live data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data_models import ObservationFrame, normalize_observation_frame


class DataSourceError(RuntimeError):
    """A source could not provide trustworthy observations."""


@dataclass(frozen=True)
class FetchContext:
    start: pd.Timestamp
    end: pd.Timestamp
    as_of: pd.Timestamp

    @classmethod
    def create(cls, start: Any, end: Any, as_of: Any | None = None) -> "FetchContext":
        def utc_naive(value: Any) -> pd.Timestamp:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            return timestamp

        start_ts = utc_naive(start)
        end_ts = utc_naive(end)
        as_of_ts = (
            utc_naive(as_of)
            if as_of is not None
            else end_ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        )
        if start_ts > end_ts:
            raise ValueError("start must not be after end")
        if as_of_ts < start_ts:
            raise ValueError("as_of must not be before start")
        return cls(start=start_ts, end=end_ts, as_of=as_of_ts)


class BaseAdapter(ABC):
    """Small interface: every source returns canonical long observations."""

    source_name: str

    @abstractmethod
    def fetch(self, start: Any, end: Any, *, as_of: Any | None = None) -> ObservationFrame:
        raise NotImplementedError

    @staticmethod
    def canonicalize(frame: pd.DataFrame) -> ObservationFrame:
        return normalize_observation_frame(frame)
