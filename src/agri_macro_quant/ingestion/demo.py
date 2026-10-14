"""Deterministic, economically linked point-in-time fixture for offline runs."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data_models import ObservationFrame, SeriesFrequency, SeriesSpec, normalize_observation_frame
from .base import BaseAdapter, FetchContext


_ANCHOR = pd.Timestamp("2000-01-03")


def _gaussian_pulse(index: pd.DatetimeIndex, center: str, width_days: float) -> np.ndarray:
    distance = (index - pd.Timestamp(center)).days.to_numpy(dtype=float)
    return np.exp(-0.5 * np.square(distance / width_days))


def _ar_component(rng: np.random.Generator, size: int, persistence: float, innovation: float) -> np.ndarray:
    values = np.zeros(size, dtype=float)
    noise = rng.normal(0.0, innovation, size)
    for position in range(1, size):
        values[position] = persistence * values[position - 1] + noise[position]
    return values


def _positive_price(
    rng: np.random.Generator,
    base: float,
    driver: np.ndarray,
    sensitivity: float,
    volatility: float,
    *,
    driver_center: float = 40.0,
    driver_scale: float = 25.0,
) -> np.ndarray:
    if driver_scale <= 0:
        raise ValueError("driver_scale must be positive")
    # Fixed economic scaling is intentional: full-sample mean/std normalization
    # would retroactively change old fixture prices whenever the end date moves.
    normalized = (driver - driver_center) / driver_scale
    deviation = np.zeros(len(driver), dtype=float)
    shocks = rng.normal(0.0, volatility, len(driver))
    for position in range(1, len(driver)):
        target = sensitivity * normalized[position]
        deviation[position] = 0.985 * deviation[position - 1] + 0.015 * target + shocks[position]
    return base * np.exp(np.clip(deviation, -1.2, 1.2))


class DemoAdapter(BaseAdapter):
    """Generate a stable synthetic research universe without network access.

    The generator begins at a fixed anchor regardless of the requested window,
    so overlapping fetches return byte-for-byte identical values for the same
    date and seed.  Historical stress pulses are synthetic analogues, not claims
    about actual releases or market closes.
    """

    source_name = "demo"

    def __init__(self, specs: Mapping[str, SeriesSpec], *, seed: int = 42) -> None:
        self.specs = dict(specs)
        self.seed = int(seed)

    def _latent_panel(self, end: pd.Timestamp) -> pd.DataFrame:
        dates = pd.date_range(_ANCHOR, end.normalize(), freq="B")

        def stream(identifier: int) -> np.random.Generator:
            return np.random.default_rng(np.random.SeedSequence([self.seed, identifier]))

        weather = (
            16.0
            + 64.0 * _gaussian_pulse(dates, "2007-08-01", 85)
            + 45.0 * _gaussian_pulse(dates, "2010-08-01", 70)
            + 72.0 * _gaussian_pulse(dates, "2012-07-15", 75)
            + 35.0 * _gaussian_pulse(dates, "2021-07-15", 90)
            + 42.0 * _gaussian_pulse(dates, "2025-06-15", 80)
            + 5.0 * np.sin(2.0 * np.pi * dates.dayofyear.to_numpy() / 365.25)
            + _ar_component(stream(1), len(dates), 0.94, 1.1)
        )
        weather = np.clip(weather, 0.0, 100.0)
        policy = (
            7.0
            + 42.0 * _gaussian_pulse(dates, "2008-04-01", 90)
            + 35.0 * _gaussian_pulse(dates, "2010-09-01", 75)
            + 26.0 * _gaussian_pulse(dates, "2020-04-01", 70)
            + 69.0 * _gaussian_pulse(dates, "2022-03-01", 150)
            + 28.0 * _gaussian_pulse(dates, "2023-08-01", 100)
            + _ar_component(stream(2), len(dates), 0.97, 0.7)
        )
        policy = np.clip(policy, 0.0, 100.0)
        scarcity = np.clip(
            0.62 * weather
            + 0.58 * policy
            + _ar_component(stream(3), len(dates), 0.98, 0.5),
            0,
            125,
        )
        energy_driver = 0.35 * weather + 0.95 * policy + 48.0 * _gaussian_pulse(dates, "2021-10-01", 170)

        wheat = _positive_price(stream(4), 5.8, scarcity, 0.34, 0.010)
        corn = _positive_price(stream(5), 4.6, 0.80 * scarcity + 0.20 * weather, 0.29, 0.009)
        soybeans = _positive_price(
            stream(6), 10.8, 0.55 * scarcity + 0.30 * weather, 0.24, 0.008
        )
        natural_gas = _positive_price(stream(7), 3.2, energy_driver, 0.50, 0.022)
        urea_latent = np.clip(
            195.0
            + 54.0 * natural_gas
            + 2.0 * policy
            + _ar_component(stream(8), len(dates), 0.97, 2.0),
            100,
            1500,
        )

        grain_revenue = 0.45 * wheat / 5.8 + 0.35 * corn / 4.6 + 0.20 * soybeans / 10.8
        margin_driver = 100.0 * grain_revenue - 0.10 * urea_latent - 0.45 * natural_gas
        cf = _positive_price(
            stream(9),
            55.0,
            margin_driver,
            0.32,
            0.012,
            driver_center=65.0,
            driver_scale=20.0,
        )
        ntr = _positive_price(
            stream(10),
            48.0,
            0.75 * margin_driver + 0.25 * scarcity,
            0.27,
            0.010,
            driver_center=58.0,
            driver_scale=20.0,
        )
        mos = _positive_price(
            stream(11),
            38.0,
            0.68 * margin_driver + 0.35 * policy,
            0.31,
            0.013,
            driver_center=48.0,
            driver_scale=20.0,
        )

        macro_noise = _ar_component(stream(12), len(dates), 0.985, 0.08)
        dxy = np.clip(94.0 + 0.075 * policy + 0.035 * scarcity + macro_noise, 70, 140)
        food_pressure = 0.55 * scarcity + 0.20 * energy_driver
        breakeven_5y = np.clip(
            1.55
            + 0.014 * food_pressure
            + _ar_component(stream(13), len(dates), 0.99, 0.012),
            -1,
            7,
        )
        us_2y = np.clip(
            1.10
            + 0.50 * breakeven_5y
            + 0.009 * policy
            + _ar_component(stream(14), len(dates), 0.995, 0.018),
            -1,
            12,
        )
        crop_condition = np.clip(
            82.0
            - 0.55 * weather
            + _ar_component(stream(15), len(dates), 0.90, 0.65),
            5,
            95,
        )
        yield_revision = np.clip(
            1.0
            - 0.105 * weather
            + _ar_component(stream(16), len(dates), 0.96, 0.15),
            -20,
            10,
        )
        exporter_stocks = np.clip(
            32.0
            - 0.105 * scarcity
            + _ar_component(stream(17), len(dates), 0.995, 0.06),
            5,
            45,
        )

        return pd.DataFrame(
            {
                "wheat": wheat,
                "corn": corn,
                "soybeans": soybeans,
                "natural_gas": natural_gas,
                "urea": urea_latent,
                "cf": cf,
                "ntr": ntr,
                "mos": mos,
                "dxy": dxy,
                "us_2y": us_2y,
                "breakeven_5y": breakeven_5y,
                "crop_condition": crop_condition,
                "yield_revision": yield_revision,
                "exporter_stocks_to_use": exporter_stocks,
                "policy_stress": policy,
                "weather_stress": weather,
            },
            index=dates,
        )

    @staticmethod
    def _sampling_dates(start: pd.Timestamp, end: pd.Timestamp, frequency: SeriesFrequency) -> pd.DatetimeIndex:
        if frequency == SeriesFrequency.DAILY:
            return pd.date_range(start, end, freq="B")
        if frequency == SeriesFrequency.WEEKLY:
            return pd.date_range(start, end, freq="W-FRI")
        if frequency == SeriesFrequency.MONTHLY:
            return pd.date_range(start, end, freq=pd.offsets.MonthEnd())
        if frequency == SeriesFrequency.QUARTERLY:
            return pd.date_range(start, end, freq=pd.offsets.QuarterEnd(startingMonth=3))
        if frequency == SeriesFrequency.ANNUAL:
            return pd.date_range(start, end, freq=pd.offsets.YearEnd())
        return pd.date_range(start, end, freq="W-FRI")

    def fetch(self, start: Any, end: Any, *, as_of: Any | None = None) -> ObservationFrame:
        context = FetchContext.create(start, end, as_of)
        panel = self._latent_panel(context.end)
        rows: list[dict[str, Any]] = []
        for series_id, spec in self.specs.items():
            if series_id not in panel.columns:
                continue
            sample_dates = self._sampling_dates(context.start, context.end, spec.frequency)
            sampled = panel[series_id].reindex(sample_dates, method="ffill")
            for observation_date, true_value in sampled.items():
                if pd.isna(true_value):
                    continue
                publication = observation_date + pd.Timedelta(days=spec.publication_lag_days)
                if spec.frequency == SeriesFrequency.DAILY and spec.publication_lag_days == 0:
                    publication += pd.Timedelta(hours=21)
                if publication > context.as_of:
                    continue
                revisable = series_id in {"yield_revision", "exporter_stocks_to_use"}
                initial_value = float(true_value)
                if revisable:
                    scale = max(abs(initial_value) * 0.025, 0.05)
                    identity = f"{self.seed}|{series_id}|{observation_date.isoformat()}".encode()
                    revision_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
                    revision_rng = np.random.default_rng(revision_seed)
                    initial_value += float(revision_rng.normal(0.0, scale))
                common = {
                    "series_id": series_id,
                    "observation_date": observation_date,
                    "publication_date": publication,
                    "effective_date": observation_date,
                    "source": self.source_name,
                    "frequency": spec.frequency.value,
                    "unit": spec.unit,
                    "source_id": spec.source_id or series_id,
                    "timestamp_quality": "synthetic_point_in_time",
                }
                rows.append(
                    {
                        **common,
                        "revision_date": publication,
                        "value": initial_value,
                        "vintage_id": "initial",
                    }
                )
                if revisable:
                    revision = publication + pd.Timedelta(days=45)
                    if revision <= context.as_of:
                        rows.append(
                            {
                                **common,
                                "revision_date": revision,
                                "value": float(true_value),
                                "vintage_id": f"revision_{revision:%Y%m%d}",
                            }
                        )
        result = normalize_observation_frame(pd.DataFrame.from_records(rows))
        result.attrs.update(
            {
                "mode": "demo",
                "seed": self.seed,
                "synthetic": True,
                "warning": "Demo values are deterministic synthetic fixtures, not investment data.",
            }
        )
        return result.sort_values(
            ["series_id", "observation_date", "revision_date"]
        ).reset_index(drop=True)
