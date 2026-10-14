"""Convert structured government actions into a release-aware policy factor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..data_models import Observation, ObservationFrame, PolicyEvent, SeriesFrequency, observations_to_frame


def policy_events_to_observations(
    events: Iterable[PolicyEvent],
    calendar: Iterable[Any] | pd.DatetimeIndex,
    *,
    half_life_days: float = 30.0,
    series_id: str = "policy_stress",
) -> ObservationFrame:
    """Aggregate signed, trade-share-weighted event stress on release dates.

    Event ``date`` is treated as both publication and effective date.  Values are
    calculated independently for each requested date using only events already
    public by that date.
    """

    event_list = list(events)
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values()
    observations: list[Observation] = []
    for date in dates:
        active = []
        for event in event_list:
            event_date = pd.Timestamp(event.date).normalize()
            age = int((date - event_date).days)
            if 0 <= age <= int(event.duration_days):
                active.append(event.stress_contribution(age, half_life_days))
        aggregate = sum(active)
        value = 50.0 + 50.0 * aggregate / (1.0 + abs(aggregate))
        observations.append(
            Observation(
                series_id=series_id,
                observation_date=date,
                publication_date=date,
                effective_date=date,
                revision_date=date,
                value=value,
                source="structured_policy_events",
                frequency=SeriesFrequency.DAILY,
                unit="score_0_100",
                vintage_id="initial",
            )
        )
    return observations_to_frame(observations)
