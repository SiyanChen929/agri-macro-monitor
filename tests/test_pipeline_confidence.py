from __future__ import annotations

import numpy as np
import pandas as pd

from agri_macro_quant.data_models import SeriesFrequency, SeriesSpec
from agri_macro_quant.pipeline import (
    _apply_asset_tradability_gate,
    _apply_point_in_time_confidence,
    _point_in_time_quality_multiplier,
    _quarantine_invalid_aligned_values,
)


def _specs() -> dict[str, SeriesSpec]:
    return {
        "critical_a": SeriesSpec(
            "critical_a",
            SeriesFrequency.DAILY,
            critical=True,
            max_staleness_days=2,
        ),
        "critical_b": SeriesSpec(
            "critical_b",
            SeriesFrequency.DAILY,
            critical=True,
            max_staleness_days=2,
        ),
        "optional": SeriesSpec(
            "optional",
            SeriesFrequency.DAILY,
            critical=False,
            max_staleness_days=2,
        ),
    }


def _signals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "ZW",
            "signal": 0.5,
            "confidence": 60.0 + np.arange(len(dates), dtype=float),
        }
    )


def test_point_in_time_quality_multiplier_requires_fresh_critical_data() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    data = pd.DataFrame(
        {
            "critical_a": [1.0, 1.0, 1.0],
            "critical_b": [2.0, 2.0, np.nan],
            "optional": [np.nan, np.nan, np.nan],
        },
        index=dates,
    )
    age = pd.DataFrame(
        {
            "critical_a": [0.0, 3.0, 0.0],
            "critical_b": [0.0, 0.0, 0.0],
            "optional": [np.nan, np.nan, np.nan],
        },
        index=dates,
    )

    multiplier = _point_in_time_quality_multiplier(data, age, _specs())

    pd.testing.assert_series_equal(
        multiplier,
        pd.Series(
            [1.0, 0.5, 0.5],
            index=dates,
            name="point_in_time_quality_multiplier",
        ),
    )


def test_historical_confidence_is_prefix_invariant_to_later_quality_failure() -> None:
    prefix_dates = pd.date_range("2024-01-02", periods=4, freq="B")
    extended_dates = pd.date_range("2024-01-02", periods=7, freq="B")
    specs = _specs()

    prefix_data = pd.DataFrame(
        {
            "critical_a": 1.0,
            "critical_b": 2.0,
            "optional": 3.0,
        },
        index=prefix_dates,
    )
    prefix_age = pd.DataFrame(0.0, index=prefix_dates, columns=prefix_data.columns)

    extended_data = prefix_data.reindex(extended_dates)
    extended_data.loc[extended_dates[4]:, "critical_a"] = 1.0
    # Endpoint degradation arrives strictly after the prefix.  It must reduce
    # new/current confidence without retroactively changing the overlap.
    extended_data.loc[extended_dates[4]:, "critical_b"] = np.nan
    extended_data.loc[extended_dates[4]:, "optional"] = 3.0
    extended_age = pd.DataFrame(0.0, index=extended_dates, columns=extended_data.columns)
    extended_age.loc[extended_dates[4]:, "critical_b"] = np.nan

    prefix = _apply_point_in_time_confidence(
        _signals(prefix_dates),
        prefix_data,
        prefix_age,
        specs,
        final_quality_multiplier=0.80,
    )
    extended = _apply_point_in_time_confidence(
        _signals(extended_dates),
        extended_data,
        extended_age,
        specs,
        final_quality_multiplier=0.0,
    )

    prefix_history = prefix.set_index("date")["backtest_confidence"]
    extended_history = extended.set_index("date")["backtest_confidence"]
    pd.testing.assert_series_equal(
        prefix_history,
        extended_history.reindex(prefix_dates),
        check_names=False,
        check_freq=False,
    )
    assert prefix["raw_confidence"].equals(_signals(prefix_dates)["confidence"])

    historical_rows = extended["date"].lt(extended_dates[-1])
    assert extended.loc[historical_rows, "current_quality_multiplier"].eq(1.0).all()
    latest = extended.loc[extended["date"].eq(extended_dates[-1])].iloc[0]
    assert latest["point_in_time_quality_multiplier"] == 0.5
    assert latest["current_quality_multiplier"] == 0.0
    assert latest["backtest_confidence"] > 0.0
    assert latest["confidence"] == 0.0


def test_impossible_critical_value_is_quarantined_and_haircuts_that_date() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    spec = SeriesSpec(
        "wheat",
        SeriesFrequency.DAILY,
        critical=True,
        min_value=0.0,
        max_value=30.0,
        max_abs_daily_return=0.35,
        max_staleness_days=2,
    )
    data = pd.DataFrame({"wheat": [5.0, -999.0, 5.1]}, index=dates)
    age = pd.DataFrame({"wheat": 0.0}, index=dates)

    clean, invalid = _quarantine_invalid_aligned_values(data, {"wheat": spec})
    multiplier = _point_in_time_quality_multiplier(clean, age, {"wheat": spec})

    assert clean["wheat"].tolist()[0] == 5.0
    assert pd.isna(clean.loc[dates[1], "wheat"])
    assert clean.loc[dates[2], "wheat"] == 5.1
    assert invalid["wheat"].tolist() == [False, True, False]
    assert multiplier.tolist() == [1.0, 0.0, 1.0]


def test_relative_value_signal_requires_both_legs_to_be_priced() -> None:
    dates = pd.bdate_range("2024-01-02", periods=25, name="date")
    returns = pd.DataFrame(
        {
            "ZW": np.linspace(0.001, 0.002, len(dates)),
            "CF": np.r_[np.linspace(0.001, 0.002, len(dates) - 1), np.nan],
        },
        index=dates,
    )
    signals = pd.DataFrame(
        {
            "date": [dates[-1]],
            "asset": ["ZW/CF"],
            "long_leg": ["ZW"],
            "short_leg": ["CF"],
            "signal": [0.8],
            "confidence": [90.0],
            "backtest_confidence": [90.0],
        }
    )

    gated = _apply_asset_tradability_gate(signals, returns)

    assert not bool(gated.iloc[0]["tradable"])
    assert "CF: latest return/mark unavailable" in gated.iloc[0]["tradability_reason"]
    assert gated.iloc[0]["confidence"] == 0.0
    assert gated.iloc[0]["backtest_confidence"] == 0.0
