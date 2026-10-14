from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from agri_macro_quant.config import (
    _validate_catalysts_config,
    _validate_data_sources_config,
    load_config,
)
from agri_macro_quant.data_models import SeriesFrequency, SeriesSpec
from agri_macro_quant.ingestion.pipeline import _fetch_live
from agri_macro_quant.pipeline import (
    _backtest_configuration,
    _factor_configuration,
    _portfolio_configuration,
)
from agri_macro_quant.risk import drawdown_risk_multiplier
from agri_macro_quant.regimes import REGIME_NAMES, regime_config_from_mapping


@pytest.mark.parametrize(
    ("section", "typo"),
    [
        ("runtime", "decision_time_uct"),
        ("quality", "failure_socre"),
        ("backtest", "rebalance_frequncy"),
        ("portfolio", "max_gross_expsoure"),
        ("models", "transition"),
    ],
)
def test_safety_critical_config_typos_are_rejected(section: str, typo: str) -> None:
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_config(overrides={section: {typo: 1}})


def test_system_version_cannot_override_package_identity() -> None:
    with pytest.raises(ValueError, match="installed package version"):
        load_config(overrides={"system": {"version": "999.0"}})


def test_series_staleness_typo_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_staleness_day"):
        SeriesSpec.from_mapping(
            "wheat",
            {"frequency": "daily", "max_staleness_day": 4},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backtest": {"transaction_cost_bps": -1.0}}, "cannot be negative"),
        ({"backtest": {"initial_capital": 0.0}}, "must be positive"),
        ({"backtest": {"rebalance_frequency": "fortnightly"}}, "daily, weekly, or monthly"),
        ({"quality": {"failure_score": 90.0, "warning_score": 80.0}}, "cannot exceed"),
        ({"features": {"minimum_periods_fraction": 1.5}}, "must be in"),
        (
            {"portfolio": {"drawdown_controls": [{"threshold": -0.1, "risk_multiplier": 1.5}]}},
            "multipliers must be between",
        ),
    ],
)
def test_unsafe_numeric_configuration_domains_are_rejected(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(overrides=overrides)


def test_series_numeric_domains_are_rejected() -> None:
    with pytest.raises(ValueError, match="min_value cannot exceed"):
        SeriesSpec(
            "x",
            SeriesFrequency.DAILY,
            min_value=10.0,
            max_value=1.0,
        )


def test_local_csv_adapter_fields_are_explicitly_allowed() -> None:
    spec = SeriesSpec.from_mapping(
        "licensed_weather",
        {
            "frequency": "weekly",
            "source": "local_csv",
            "location": "data/weather.csv",
            "date_column": "release_date",
            "value_column": "score",
        },
    )

    assert spec.source == "local_csv"


@pytest.mark.parametrize(
    "overrides",
    [
        {"factors": {"per_factor": {}}},
        {"factors": {"factor_settings": {}}},
        {"risk": {"input_type": "returns"}},
        {"risk": {"drawdown_controls": []}},
        {"risk": {"thresholds": []}},
        {"models": {"statistical_regime": {"random_seed": 9}}},
        {"models": {"transitions": {"random_seed": 9}}},
    ],
)
def test_accepted_but_unused_operational_keys_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_config(overrides=overrides)


def test_factor_include_raw_is_forwarded_to_factor_engine() -> None:
    config = load_config(overrides={"factors": {"include_raw": True}})

    assert _factor_configuration(config)["include_raw"] is True


def test_explicit_empty_drawdown_schedule_disables_both_backtest_and_current_control() -> None:
    config = load_config(overrides={"portfolio": {"drawdown_controls": []}})
    backtest_config = _backtest_configuration(
        config,
        _portfolio_configuration(config),
    )

    assert backtest_config["drawdown_controls"] == []
    assert drawdown_risk_multiplier(-0.50, backtest_config) == 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"soruces": {}}), "unknown configuration key"),
        (
            lambda value: value["sources"]["yahoo"].update({"timeot_seconds": 2}),
            "timeot_seconds",
        ),
        (
            lambda value: value["sources"]["yahoo"].update({"adapter": "fred"}),
            "adapter must be 'yahoo_chart'",
        ),
        (
            lambda value: value["sources"].update(
                {"yahooo": {"adapter": "yahoo_chart"}}
            ),
            "unsupported provider 'yahooo'",
        ),
        (
            lambda value: value["series"]["wheat"].update({"source": "yahooo"}),
            "unsupported provider 'yahooo'",
        ),
    ],
)
def test_data_source_schema_and_provider_consistency_are_strict(mutation, message: str) -> None:
    payload = deepcopy(load_config().data_sources)
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        _validate_data_sources_config(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"weights": {}, "temperatur": 1.0}, "unknown top-level"),
        ({"weights": {"not_a_regime": {"dxy": 1.0}}}, "unknown regime"),
        (
            {"weights": {REGIME_NAMES[0]: {"weather_stres": 1.0}}},
            "unknown factor",
        ),
        ({"intercepts": {"not_a_regime": 0.0}}, "unknown regime"),
        ({"labels": {"not_a_regime": "bad"}}, "unknown regime"),
        (
            {
                "regimes": {
                    REGIME_NAMES[0]: {
                        "weights": {"weather_stress": 1.0},
                        "weigths": {},
                    }
                }
            },
            "unknown key",
        ),
        ({"model": {"temperatur": 1.0}}, "model contains unknown key"),
    ],
)
def test_regime_configuration_rejects_unknown_keys(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        regime_config_from_mapping(payload)


def test_shipped_and_nested_regime_configuration_forms_are_consumed() -> None:
    shipped = regime_config_from_mapping(load_config().regime_thresholds)
    assert set(shipped["weights"]) == set(REGIME_NAMES)

    nested = regime_config_from_mapping(
        {
            "model": {"temperature": 1.5},
            "regimes": {
                "weather_shock": {
                    "weights": {"weather_stress_z": 1.25},
                    "intercept": -0.2,
                    "label": "Weather test",
                }
            },
        }
    )
    canonical = "weather_crop_supply_shock"
    assert nested["weights"][canonical] == {"weather_stress": 1.25}
    assert nested["intercepts"][canonical] == pytest.approx(-0.2)
    assert nested["labels"][canonical] == "Weather test"
    assert nested["temperature"] == pytest.approx(1.5)


def test_csv_alias_honors_local_csv_provider_disable_flag() -> None:
    base = load_config()
    payload = deepcopy(base.data_sources)
    payload["sources"]["local_csv"]["enabled"] = False
    spec = SeriesSpec(
        "licensed_weather",
        SeriesFrequency.DAILY,
        source="csv",
        source_id="unused.csv",
        critical=True,
    )
    config = replace(
        base,
        data_sources=payload,
        series={"licensed_weather": spec},
    )

    result = _fetch_live(
        config,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-05"),
    )

    assert result.empty
    assert result.attrs["ingestion_issues"][0]["code"] == "source_disabled"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"calendar_type": "verified_official"}), "cannot be relabelled"),
        (lambda value: value.update({"business_day_adjustment": "folowing"}), "business_day_adjustment"),
        (lambda value: value["recurring"][0].update({"cadence": "monthy"}), "cadence is unsupported"),
        (lambda value: value["recurring"][0].update({"expected_importance": "urgent"}), "importance is unsupported"),
        (lambda value: value["recurring"][0].update({"affected_regime": "weather_shok"}), "canonical regime"),
    ],
)
def test_catalyst_fixture_schema_cannot_silently_drop_or_upgrade_events(
    mutation,
    message: str,
) -> None:
    payload = deepcopy(load_config().catalysts)
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        _validate_catalysts_config(payload)
