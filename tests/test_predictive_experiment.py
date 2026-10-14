from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from agri_macro_quant.research.predictive_experiment import (
    PredictiveExperimentConfig,
    _metrics_table,
    _ridge_fit_predict,
    _select_development_model,
    asset_crop_year,
    build_daily_prediction_panel,
    development_selection_plan,
    make_crop_year_folds,
    run_predictive_experiment,
    sample_weekly_panel,
)


def _price_panel(
    start: str,
    end: str,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    state = np.zeros(len(index), dtype=float)
    shocks = rng.normal(0.0, 0.20, len(index))
    for position in range(1, len(index)):
        state[position] = 0.97 * state[position - 1] + shocks[position]
    prices: dict[str, np.ndarray] = {}
    for number, asset in enumerate(("ZW", "ZC", "ZS")):
        noise = rng.normal(0.0, 0.004, len(index))
        daily_return = 0.00005 + 0.00045 * np.roll(state, 1) + noise
        daily_return[0] = 0.0
        prices[asset] = (100.0 + 10.0 * number) * np.exp(np.cumsum(daily_return))
    usda = pd.DataFrame(
        {
            "yield_revision": state,
            "stocks_to_use": -0.5 * state + rng.normal(0.0, 0.1, len(index)),
        },
        index=index,
    )
    weather = pd.DataFrame(
        {
            "heat_stress": state + rng.normal(0.0, 0.1, len(index)),
            "rain_deficit": 0.25 * state + rng.normal(0.0, 0.2, len(index)),
        },
        index=index,
    )
    weather.iloc[::37, 1] = np.nan
    return pd.DataFrame(prices, index=index), usda, weather


def test_target_is_next_sessions_and_proxy_roll_gap_is_neutralized() -> None:
    index = pd.bdate_range("2020-03-09", periods=28)
    prices = pd.DataFrame(
        {
            "ZW": np.linspace(100.0, 105.0, len(index)),
            "ZC": np.linspace(110.0, 114.0, len(index)),
            "ZS": np.linspace(120.0, 123.0, len(index)),
        },
        index=index,
    )
    jump_date = pd.Timestamp("2020-03-16")
    prices.loc[jump_date:, "ZW"] *= 1.20
    config = PredictiveExperimentConfig(horizon_sessions=2)
    panel, _ = build_daily_prediction_panel(
        prices,
        config=config,
        proxy_input=True,
    )
    wheat = panel.xs("ZW", level="asset")

    assert bool(wheat.loc[jump_date, "proxy_roll_gap_flag"])
    prior = index[index.get_loc(jump_date) - 1]
    assert wheat.loc[jump_date, "price"] == pytest.approx(wheat.loc[prior, "price"])
    assert wheat.loc[jump_date, "raw_price"] > wheat.loc[jump_date, "price"]
    first = index[0]
    target_start = index[1]
    target_end = index[3]
    assert wheat.loc[first, "target_start_date"] == target_start
    assert wheat.loc[first, "target_end_date"] == target_end
    assert wheat.loc[first, "target_return_21d"] == pytest.approx(
        wheat.loc[target_end, "price"] / wheat.loc[target_start, "price"] - 1.0
    )

    raw_config = PredictiveExperimentConfig(
        horizon_sessions=2,
        neutralize_proxy_roll_gaps=False,
    )
    raw_panel, _ = build_daily_prediction_panel(
        prices,
        config=raw_config,
        proxy_input=True,
    )
    raw_wheat = raw_panel.xs("ZW", level="asset")
    assert bool(raw_wheat.loc[jump_date, "proxy_roll_gap_flag"])
    assert raw_wheat.loc[jump_date, "price"] == raw_wheat.loc[jump_date, "raw_price"]


def test_proxy_repair_keeps_large_pre_anchor_market_move() -> None:
    index = pd.bdate_range("2022-03-01", "2022-03-22")
    prices = pd.DataFrame(
        {asset: np.linspace(100.0, 103.0, len(index)) for asset in ("ZW", "ZC", "ZS")},
        index=index,
    )
    genuine_shock = pd.Timestamp("2022-03-08")
    prices.loc[genuine_shock:, "ZW"] *= 0.88
    panel, _ = build_daily_prediction_panel(prices, proxy_input=True)
    wheat = panel.xs("ZW", level="asset")

    assert not bool(wheat.loc[genuine_shock, "proxy_roll_gap_flag"])
    assert wheat.loc[genuine_shock, "price"] < wheat.loc[index[index.get_loc(genuine_shock) - 1], "price"]
    assert bool(wheat.loc[pd.Timestamp("2022-03-15"), "proxy_roll_gap_flag"])


def test_target_shifts_over_valid_asset_sessions() -> None:
    index = pd.bdate_range("2020-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "ZW": [100.0, np.nan, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "ZC": np.linspace(100.0, 107.0, len(index)),
            "ZS": np.linspace(100.0, 107.0, len(index)),
        },
        index=index,
    )
    config = PredictiveExperimentConfig(horizon_sessions=2)
    panel, _ = build_daily_prediction_panel(prices, config=config)
    wheat = panel.xs("ZW", level="asset")

    assert wheat.loc[index[0], "target_start_date"] == index[2]
    assert wheat.loc[index[0], "target_end_date"] == index[4]
    assert wheat.loc[index[0], "target_return_21d"] == pytest.approx(
        prices.loc[index[4], "ZW"] / prices.loc[index[2], "ZW"] - 1.0
    )


def test_crop_year_folds_are_purged_and_reserve_latest_complete_years() -> None:
    prices, _, _ = _price_panel("2000-01-03", "2021-12-31")
    config = PredictiveExperimentConfig(
        minimum_train_crop_years=5,
        holdout_crop_years=3,
        minimum_weekly_observations_per_crop_year=40,
        alphas=(0.1, 1.0),
    )
    daily, _ = build_daily_prediction_panel(prices, config=config)
    weekly = sample_weekly_panel(daily, config)
    folds = make_crop_year_folds(weekly, "ZC", config)
    corn = weekly.xs("ZC", level="asset")

    assert asset_crop_year("2020-05-31", "ZW") == 2019
    assert asset_crop_year("2020-06-01", "ZW") == 2020
    assert asset_crop_year("2020-08-31", "ZC") == 2019
    assert asset_crop_year("2020-09-01", "ZC") == 2020
    assert tuple(fold.test_crop_year for fold in folds[-3:]) == tuple(
        sorted(fold.test_crop_year for fold in folds)[-3:]
    )
    assert all(fold.is_holdout for fold in folds[-3:])
    assert any(not fold.is_holdout for fold in folds)
    for fold in folds:
        train = corn.reindex(pd.DatetimeIndex(fold.train_dates))
        assert len(fold.train_crop_years) >= 5
        assert pd.to_datetime(train["target_end_date"]).max() < fold.purge_cutoff


def test_development_selection_plan_uses_earliest_cross_asset_holdout() -> None:
    prices, usda, weather = _price_panel("2010-01-01", "2025-12-31", seed=19)
    config = PredictiveExperimentConfig(
        minimum_train_crop_years=4,
        holdout_crop_years=2,
        minimum_weekly_observations_per_crop_year=40,
    )
    daily, _ = build_daily_prediction_panel(
        prices,
        usda_features=usda,
        weather_features=weather,
        config=config,
    )
    weekly = sample_weekly_panel(daily, config)
    folds = tuple(
        fold
        for asset in config.assets
        for fold in make_crop_year_folds(weekly, asset, config)
    )
    cutoff, eligible = development_selection_plan(weekly, folds, config)

    earliest_holdout = min(min(fold.test_dates) for fold in folds if fold.is_holdout)
    assert cutoff == pd.Timestamp(earliest_holdout) - pd.offsets.BDay(
        config.embargo_business_days
    )
    for asset, year in eligible:
        fold = next(
            item for item in folds if item.asset == asset and item.test_crop_year == year
        )
        frame = weekly.xs(asset, level="asset").reindex(fold.test_dates)
        assert pd.to_datetime(frame["target_end_date"]).max() < cutoff


def test_ridge_preprocessing_uses_training_distribution_only() -> None:
    index = pd.bdate_range("2010-01-01", periods=12)
    train = pd.DataFrame(
        {
            "feature": np.linspace(-1.0, 1.0, 10),
            "target_return_21d": np.linspace(-0.02, 0.02, 10),
        },
        index=index[:10],
    )
    test = pd.DataFrame({"feature": [0.25, 0.5]}, index=index[10:])
    mutated = test.copy()
    mutated.iloc[1, 0] = 1.0e12
    config = PredictiveExperimentConfig(
        minimum_feature_coverage=0.5,
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    original, _, _, _ = _ridge_fit_predict(train, test, ["feature"], 1.0, config)
    changed, _, _, _ = _ridge_fit_predict(train, mutated, ["feature"], 1.0, config)
    assert changed[0] == pytest.approx(original[0])


def test_development_selection_requires_every_preregistered_gate() -> None:
    records: list[dict[str, object]] = []
    assets = ("ZW", "ZC", "ZS")
    years = (2010, 2011)
    folds = [(asset, year) for asset in assets for year in years]
    for position, (asset, year) in enumerate(folds):
        common = {
            "date": pd.Timestamp(year=year, month=10, day=1),
            "asset": asset,
            "crop_year": year,
            "split": "development",
            "actual": 0.0,
        }
        records.append({**common, "model": "ridge_core", "prediction": 1.0})
        # Lower pooled MSE, but only wheat and two of six folds improve.
        records.append(
            {
                **common,
                "model": "ridge_core_usda",
                "prediction": 0.5 if asset == "ZW" else 1.05,
            }
        )
        # Two assets improve, but only three of six folds improve (not a majority).
        weather_improves = position in {0, 1, 2}
        records.append(
            {
                **common,
                "model": "ridge_core_weather",
                "prediction": 0.0 if weather_improves else 1.1,
            }
        )
        # Five of six folds and all three assets improve.
        records.append(
            {
                **common,
                "model": "ridge_core_usda_weather",
                "prediction": 0.5 if position < 5 else 1.1,
            }
        )
    development = pd.DataFrame(records)
    all_models = (
        "ridge_core",
        "ridge_core_usda",
        "ridge_core_weather",
        "ridge_core_usda_weather",
    )

    assert _select_development_model(development, all_models) == (
        "ridge_core_usda_weather"
    )
    assert _select_development_model(development, all_models[:-1]) == "ridge_core"

    # Deliberately extreme holdout rows cannot alter the development decision.
    holdout = development.copy()
    holdout["split"] = "holdout"
    holdout.loc[holdout["model"].eq("ridge_core"), "prediction"] = 1000.0
    holdout.loc[
        holdout["model"].eq("ridge_core_usda_weather"), "prediction"
    ] = 0.0
    combined = pd.concat([development, holdout], ignore_index=True)
    assert _select_development_model(combined, all_models) == (
        "ridge_core_usda_weather"
    )


def test_oos_r2_uses_matching_historical_mean_group() -> None:
    rows: list[dict[str, object]] = []
    for asset, year, baseline_error in (
        ("ZW", 2010, 1.0),
        ("ZW", 2011, 2.0),
        ("ZC", 2010, 3.0),
        ("ZC", 2011, 4.0),
    ):
        common = {
            "split": "development",
            "asset": asset,
            "crop_year": year,
            "actual": 0.0,
        }
        rows.append(
            {
                **common,
                "model": "historical_mean",
                "prediction": baseline_error,
            }
        )
        rows.append(
            {
                **common,
                "model": "ridge_core",
                "prediction": baseline_error / 2.0,
            }
        )
    predictions = pd.DataFrame(rows)

    by_asset = _metrics_table(predictions, ["split", "model", "asset"])
    by_fold = _metrics_table(
        predictions, ["split", "model", "asset", "crop_year"]
    )
    assert np.allclose(
        by_asset.loc[
            by_asset["model"].eq("ridge_core"),
            "oos_r2_vs_historical_mean",
        ],
        0.75,
    )
    assert np.allclose(
        by_fold.loc[
            by_fold["model"].eq("ridge_core"),
            "oos_r2_vs_historical_mean",
        ],
        0.75,
    )


def test_experiment_selects_on_development_then_opens_only_selected_holdout() -> None:
    prices, usda, weather = _price_panel("2004-01-02", "2018-12-31")
    config = PredictiveExperimentConfig(
        minimum_train_crop_years=3,
        holdout_crop_years=2,
        minimum_weekly_observations_per_crop_year=35,
        alphas=(0.1, 1.0),
    )
    result = run_predictive_experiment(
        prices,
        usda_features=usda,
        weather_features=weather,
        config=config,
        proxy_input=True,
    )

    assert result.proxy_input is True
    assert result.selected_model.startswith("ridge_")
    development_ridge = set(
        result.predictions.loc[
            result.predictions["split"].eq("development")
            & result.predictions["model"].str.startswith("ridge_"),
            "model",
        ]
    )
    assert development_ridge == {
        "ridge_core",
        "ridge_core_usda",
        "ridge_core_weather",
        "ridge_core_usda_weather",
    }
    holdout_ridge = set(
        result.predictions.loc[
            result.predictions["split"].eq("holdout")
            & result.predictions["model"].str.startswith("ridge_"),
            "model",
        ]
    )
    assert holdout_ridge == {"ridge_core", result.selected_model}
    holdout_models = set(
        result.predictions.loc[
            result.predictions["split"].eq("holdout"), "model"
        ]
    )
    assert holdout_models == {
        "historical_mean",
        "seasonality",
        "momentum_60",
        "ridge_core",
        result.selected_model,
    }
    assert all(len(years) == 2 for years in result.holdout_crop_years.values())
    assert not result.coefficients.empty
    assert not result.hyperparameters.empty
    assert not result.metrics_overall.empty
    assert not result.metrics_by_asset.empty
    assert not result.metrics_by_crop_year.empty
    assert np.isfinite(result.predictions[["actual", "prediction"]]).all().all()

    expected = _select_development_model(
        result.predictions,
        tuple(development_ridge),
    )
    assert result.selected_model == expected
    with pytest.raises(FrozenInstanceError):
        result.selected_model = "ridge_core"  # type: ignore[misc]
