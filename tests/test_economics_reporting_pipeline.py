from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_macro_quant.backtest import run_backtest
from agri_macro_quant.data_models import PolicyEvent
from agri_macro_quant.economics import calculate_exportable_surplus, policy_stress_series
from agri_macro_quant.pipeline import run_pipeline
from agri_macro_quant.regimes import score_regimes
from agri_macro_quant.reporting import _plain, _what_changed, generate_reports
from agri_macro_quant.signals import SIGNAL_COLUMNS, generate_trade_signals
from agri_macro_quant.transitions import estimate_transition_probabilities


def test_exportable_surplus_uses_only_major_exporters_and_correct_denominator() -> None:
    fundamentals = pd.DataFrame(
        [
            {
                "date": "2024-06-01",
                "country": "Exporter A",
                "commodity": "wheat",
                "production": 100,
                "beginning_stocks": 10,
                "imports": 5,
                "domestic_consumption": 60,
                "required_ending_stocks": 10,
                "ending_stocks": 12,
                "exports": 30,
                "is_major_exporter": True,
            },
            {
                "date": "2024-06-01",
                "country": "Exporter B",
                "commodity": "wheat",
                "production": 80,
                "beginning_stocks": 5,
                "imports": 0,
                "domestic_consumption": 50,
                "required_ending_stocks": 5,
                "ending_stocks": 8,
                "exports": 20,
                "is_major_exporter": True,
            },
            {
                "date": "2024-06-01",
                "country": "Large non-exporter",
                "commodity": "wheat",
                "production": 1_000,
                "beginning_stocks": 500,
                "imports": 100,
                "domestic_consumption": 900,
                "required_ending_stocks": 300,
                "ending_stocks": 400,
                "exports": 2,
                "is_major_exporter": False,
            },
        ]
    )

    result = calculate_exportable_surplus(fundamentals)
    row = result.iloc[0]

    assert len(result) == 1
    assert row["global_exportable_surplus"] == 75.0
    assert row["exporter_ending_stocks"] == 20.0
    assert row["exporter_domestic_use"] == 110.0
    assert row["exporter_exports"] == 50.0
    assert row["exporter_count"] == 2
    assert row["exporter_stocks_to_use"] == pytest.approx(12.5)


def test_policy_stress_is_causal_trade_weighted_decayed_and_duration_limited() -> None:
    event = PolicyEvent(
        date=pd.Timestamp("2024-01-02"),
        country="Exporter A",
        commodity="wheat",
        policy_type="export restriction",
        direction=1.0,
        magnitude=2.0,
        duration_days=2,
        confidence=0.5,
        source="test",
        expected_domestic_effect="lower domestic price",
        expected_global_effect="higher global price",
        trade_share=0.25,
        importance=0.8,
    )
    calendar = pd.date_range("2024-01-01", "2024-01-05", freq="D")

    result = policy_stress_series([event], calendar, half_life_days=2.0)

    initial = 1.0 * 2.0 * 0.5 * 0.25 * 0.8
    expected = pd.Series(
        [0.0, initial, initial / np.sqrt(2.0), initial / 2.0, 0.0],
        index=calendar,
        name="policy_stress",
    )
    pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    assert event.stress_contribution(age_days=2, half_life_days=2.0) == pytest.approx(
        initial / 2.0
    )


def test_what_changed_attributes_the_change_not_the_largest_current_level() -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    regime = "weather_crop_supply_shock"
    scores = pd.DataFrame({regime: [50.0, 60.0]}, index=dates)
    contributions = pd.DataFrame(
        [
            {
                "date": dates[0],
                "regime": regime,
                "factor": "large_static_driver",
                "score_point_contribution": 8.0,
            },
            {
                "date": dates[1],
                "regime": regime,
                "factor": "large_static_driver",
                "score_point_contribution": 8.0,
            },
            {
                "date": dates[0],
                "regime": regime,
                "factor": "changing_driver",
                "score_point_contribution": 1.0,
            },
            {
                "date": dates[1],
                "regime": regime,
                "factor": "changing_driver",
                "score_point_contribution": 5.0,
            },
        ]
    )

    statement = _what_changed(scores, contributions, lookback=1)[0]

    assert "largest measured contribution change: changing driver (+4.0 score points)" in statement
    assert "large static driver" not in statement


def test_reporting_writes_human_machine_and_chart_artifacts(
    tmp_path: Path,
    market_frame: pd.DataFrame,
    standardized_factors: pd.DataFrame,
) -> None:
    index = standardized_factors.index
    market = market_frame.tail(len(index)).copy()
    market.index = index
    regimes = score_regimes(standardized_factors, {"min_coverage": 1.0})
    transitions = estimate_transition_probabilities(
        regimes,
        config={"horizons": {"1m": 5}, "smoothing": 1.0},
    )
    signals = generate_trade_signals(
        standardized_factors,
        regimes,
        transition_probs=transitions,
    )
    returns = pd.DataFrame(
        {
            "ZW": market["wheat"].pct_change(fill_method=None),
            "ZC": market["corn"].pct_change(fill_method=None),
        },
        index=index,
    )
    target = pd.DataFrame({"ZW": 0.20, "ZC": -0.15}, index=index)
    backtest = run_backtest(
        returns=returns,
        target_weights=target,
        config={
            "execution_lag": 1,
            "transaction_cost_bps": 1.0,
            "slippage_bps": 0.0,
            "futures_roll_cost_bps": 0.0,
            "amortize_roll_cost": False,
            "apply_drawdown_control": False,
        },
    )
    output = tmp_path / "report"

    artifacts = generate_reports(
        output,
        as_of=index[-1],
        system_version="test-version",
        market=market,
        factors=standardized_factors,
        regime_scores=regimes.scores,
        contributions=regimes,
        transition_probabilities=transitions,
        signals=signals,
        portfolio_weights=pd.Series({"ZW": 0.20, "ZC": -0.15}),
        risk_report={"gross_exposure": 0.35, "net_exposure": 0.05},
        backtest=backtest,
        scenarios=pd.DataFrame(
            {"scenario": ["weather shock"], "portfolio_return": [-0.03]}
        ),
        data_quality={"score": 98.0, "status": "healthy", "issues": []},
        source_label="Deterministic test fixture",
        run_metadata={"point_in_time_safe": True, "execution_lag": 1},
        model_health={"status": "healthy", "warnings": [], "regime_stability": 0.9},
    )

    assert {"html", "markdown", "technical_markdown", "json", "charts"} == set(artifacts)
    for name in ("html", "markdown", "technical_markdown", "json"):
        assert artifacts[name].is_file() and artifacts[name].stat().st_size > 0
    assert artifacts["charts"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts["charts"])

    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert payload["date"] == str(index[-1].date())
    assert payload["system_version"] == "test-version"
    assert payload["dominant_regime"]
    assert 0.0 <= payload["confidence"] <= 100.0
    assert "not a probability" in payload["confidence_method"]
    assert len(payload["regime_scores"]) == 7
    assert payload["run_metadata"]["point_in_time_safe"] is True
    assert payload["signals"]
    assert payload["charts"]

    html = artifacts["html"].read_text(encoding="utf-8")
    markdown = artifacts["markdown"].read_text(encoding="utf-8")
    technical = artifacts["technical_markdown"].read_text(encoding="utf-8")
    assert "Research output, not automatic execution" in html
    assert "Executive Summary" in html
    assert "What the Market May Be Missing" in markdown
    assert "Data Quality and Model Health" in technical


def test_report_json_normalizer_rejects_nonfinite_machine_scalars() -> None:
    assert _plain({"probability": np.nan, "score": np.inf}) == {
        "probability": None,
        "score": None,
    }


def test_short_offline_demo_pipeline_runs_end_to_end_and_persists_contracts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo_run"

    result = run_pipeline(
        mode="demo",
        start="2024-01-02",
        as_of="2024-06-28",
        output_dir=output,
        run_statistical=False,
        record_research_log=False,
    )

    assert result.output_dir == output.resolve()
    assert not result.observations.empty
    assert not result.market.empty
    assert result.market.index.min() < pd.Timestamp("2024-01-02")
    assert not result.features.empty
    assert any(str(column).endswith("__data_age_days") for column in result.features)
    assert not result.factors.empty
    assert not result.regimes.scores.empty
    assert result.statistical_regimes is None
    assert set(SIGNAL_COLUMNS).issubset(result.signals.columns)
    assert not result.signals.empty
    assert result.signals["data_mode"].eq("demo_synthetic").all()
    assert np.isfinite(result.backtest.equity_curve).all()
    assert result.backtest.returns.index.min() == pd.Timestamp("2024-01-02")
    assert result.manifest["mode"] == "demo"
    assert result.manifest["quality"]["status"] == result.quality.status
    assert not result.lead_lag.granger_tests.empty
    assert result.research_questions.set_index("question").loc["Q10", "status"] == "tested"
    assert np.isclose(
        result.risk["simulated_strategy_drawdown"],
        result.backtest.drawdown.iloc[-1],
    )
    assert result.risk["post_control_gross_exposure"] <= result.risk["pre_control_gross_exposure"] + 1e-12

    for history in result.transitions.forecast_history.values():
        np.testing.assert_allclose(history.sum(axis=1), 1.0, rtol=0, atol=1e-10)

    required_paths = [
        "report.html",
        "report.md",
        "technical_report.md",
        "monitor.json",
        "manifest.json",
        "data_quality.json",
        "configuration_snapshot.json",
        "data/raw/observations.csv.gz",
        "data/processed/market_point_in_time.csv.gz",
        "data/processed/signals.csv.gz",
        "data/processed/current_portfolio.csv",
        "data/processed/backtest_metrics.json",
        "diagnostics/regime_contributions.csv.gz",
        "diagnostics/transition_matrix_1m.csv",
        "diagnostics/lead_lag_granger_tests.csv",
        "diagnostics/research_question_status.csv",
        "diagnostics/uncontrolled_current_portfolio.csv",
    ]
    for relative in required_paths:
        path = output / relative
        assert path.is_file() and path.stat().st_size > 0, relative

    monitor = json.loads((output / "monitor.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert monitor["run_metadata"]["point_in_time_safe"] is True
    assert monitor["run_metadata"]["execution_lag"] == 1
    assert 0.0 <= monitor["expected_next_probability"] <= 1.0
    assert np.isclose(
        monitor["expected_next_probability_percent"],
        100.0 * monitor["expected_next_probability"],
    )
    assert manifest["effective_runtime"]["evaluation_start"].startswith("2024-01-02")
    assert manifest["effective_runtime"]["fetch_start"] < manifest["effective_runtime"]["evaluation_start"]
    assert manifest["outputs"]["report_html"] == "report.html"
    assert manifest["outputs"]["chart_count"] >= 8
    assert len(list((output / "charts").glob("*.png"))) == manifest["outputs"]["chart_count"]
