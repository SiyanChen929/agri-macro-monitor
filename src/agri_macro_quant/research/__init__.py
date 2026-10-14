"""Research utilities for event and lead/lag validation."""

from .event_study import EventStudyResult, event_study, run_event_study
from .lead_lag import (
    ASSOCIATION_DISCLAIMER,
    LeadLagAnalysisResult,
    cross_correlation_scan,
    distributed_lag_regression,
    granger_causality_test,
    run_lead_lag_analysis,
)

__all__ = [
    "ASSOCIATION_DISCLAIMER",
    "EventStudyResult",
    "LeadLagAnalysisResult",
    "cross_correlation_scan",
    "distributed_lag_regression",
    "event_study",
    "granger_causality_test",
    "run_event_study",
    "run_lead_lag_analysis",
]
