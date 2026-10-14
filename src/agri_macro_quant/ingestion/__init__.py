"""Canonical ingestion API."""

from .adapters import CSVAdapter, FREDAdapter, YahooChartAdapter
from .base import BaseAdapter, DataSourceError, FetchContext
from .catalysts import build_catalyst_calendar
from .demo import DemoAdapter
from .pipeline import ingest_all, load_demo_data
from .policy import policy_events_to_observations
from .snapshot import load_snapshot, save_snapshot

__all__ = [
    "BaseAdapter",
    "CSVAdapter",
    "DataSourceError",
    "DemoAdapter",
    "FREDAdapter",
    "FetchContext",
    "YahooChartAdapter",
    "build_catalyst_calendar",
    "ingest_all",
    "load_demo_data",
    "load_snapshot",
    "policy_events_to_observations",
    "save_snapshot",
]

