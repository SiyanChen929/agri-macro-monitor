# AGRICULTURAL MACRO REGIME MONITOR

**Date:** 2026-08-29  
**System Version:** 0.1.0  
**Data:** DEMO — deterministic synthetic point-in-time fixture; not investment data

**DATA QUALITY WARNING — confidence is haircutted; review the quality issues before using any research candidate.**

## Current Dominant Regime

**Supply Response / Mean Reversion** — 98.3/100, heuristic conviction 75% (uncalibrated; not a probability), one-week change -0.4, Stable →.

Expected next regime: **Supply Response / Mean Reversion** (82.5%).

## Executive Summary

The dominant measured regime is Supply Response / Mean Reversion, with a score of 98.3/100 and 75% heuristic conviction after data-quality adjustment; this is not a calibrated probability. It changed -0.4 points over the latest five-session window; the largest current contribution is yield revision. The one-month transition model assigns its highest probability to Supply Response / Mean Reversion (82.5%). Trade candidates are ranked from standardized signals, regime agreement, and data quality; they are research outputs rather than execution instructions. The most important model risk is 9 changes exceed robust z=8. Simple benchmark time_series_momentum achieved Sharpe 0.56 versus -0.19 for the regime strategy in this sample.

## Regime Dashboard

| Regime                                  |   Score |   Δ 1D |   Δ 1W | Trend           |
|:----------------------------------------|--------:|-------:|-------:|:----------------|
| Supply Response / Mean Reversion        |    98.3 |   -0.1 |   -0.4 | Stable →        |
| Government / Trade Policy Amplification |    58.5 |   -0.1 |    0.8 | Stable →        |
| Agricultural Input Expansion            |    20.8 |   -2.6 |    5.3 | Strengthening ↑ |
| Monetary Tightening / Stagflation       |    16.1 |   -3.5 |    5.3 | Strengthening ↑ |
| Inventory / Food Security Stress        |     9.4 |   -0.4 |    1.4 | Strengthening ↑ |
| Food Inflation Transmission             |     5.4 |   -0.7 |    1.9 | Strengthening ↑ |
| Weather / Crop Supply Shock             |     0.3 |   -0.0 |    0.1 | Stable →        |

## Current Causal Chain

Measured: supply-response inputs produced the leading score → Model pathway: strong crop economics could improve planting incentives → Model pathway: better incentives could expand input use or acreage → Model pathway: favorable yields could rebuild production and inventories → Hypothesis to monitor: greater supply may create mean-reversion pressure

## What Changed

1. Monetary Tightening / Stagflation moved from 10.8 to 16.1 (+5.3); largest measured contribution change: dxy (+5.0 score points).
2. Agricultural Input Expansion moved from 15.6 to 20.8 (+5.3); largest measured contribution change: corn urea (+3.3 score points).
3. Food Inflation Transmission moved from 3.5 to 5.4 (+1.9); largest measured contribution change: wheat momentum (+1.9 score points).
4. Inventory / Food Security Stress moved from 8.0 to 9.4 (+1.4); largest measured contribution change: wheat momentum (+2.2 score points).
5. Government / Trade Policy Amplification moved from 57.7 to 58.5 (+0.8); largest measured contribution change: policy stress (-2.3 score points).

## Best Research Trades

|   rank | asset   | direction    |   strength |   confidence | expected_horizon   | economic_driver               |
|-------:|:--------|:-------------|-----------:|-------------:|:-------------------|:------------------------------|
|      1 | ZC      | short        |   0.942718 |      92.2933 | 2-8 weeks          | weather_stress                |
|      2 | NTR/MOS | long_spread  |   0.922238 |      90.7649 | 2-8 weeks          | potash versus phosphate       |
|      3 | ZC/CF   | short_spread |   0.910422 |      89.6966 | 2-8 weeks          | corn versus nitrogen producer |
|      4 | ZS      | short        |   0.917062 |      88.5445 | 2-8 weeks          | weather_stress                |
|      5 | ZW      | short        |   0.933914 |      86.332  | 2-6 weeks          | weather_stress                |

1. ZC (short) — Thesis: weather_stress. Confirmation: the signed signal and regime evidence persist. Invalidation: crop conditions deteriorate, yield expectations fall, or inventories tighten. Main risk: the measured relationship may be sample-specific or already priced.
2. NTR/MOS (long_spread) — Thesis: potash versus phosphate. Confirmation: the signed signal and regime evidence persist. Invalidation: the return spread divergence reverses or loses statistical significance. Main risk: the measured relationship may be sample-specific or already priced.
3. ZC/CF (short_spread) — Thesis: corn versus nitrogen producer. Confirmation: the signed signal and regime evidence persist. Invalidation: the beta neutral divergence reverses or loses statistical significance. Main risk: the measured relationship may be sample-specific or already priced.

## What the Market May Be Missing

1. Model inference: weak input affordability is a counterweight to bullish fertilizer-demand narratives.
2. Model inference: exporter inventory conditions do not currently support a scarcity premium.
3. Model inference: wheat performance relative to dollar pressure suggests the physical signal may differ from the macro signal.

## Physical Market

Aligned decision-panel values below may be carried from an earlier release; see the series-freshness table for actual observation and availability dates.

|                        |   2026-08-27 |   2026-08-28 |
|:-----------------------|-------------:|-------------:|
| weather_stress         |    7.55637   |    7.55637   |
| crop_condition         |   74.7189    |   74.7189    |
| yield_revision         |   -0.0477421 |   -0.0477421 |
| exporter_stocks_to_use |   31.7578    |   31.7578    |
| wheat                  |    3.94143   |    3.90123   |
| corn                   |    3.14743   |    3.10335   |
| soybeans               |    7.37531   |    7.42003   |

### Series Freshness

| series_id              | source   | unit                   | last_observation_date   | last_available_date   |   data_age_days |   max_staleness_days | stale   |
|:-----------------------|:---------|:-----------------------|:------------------------|:----------------------|----------------:|---------------------:|:--------|
| wheat                  | demo     | USD/bushel             | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| corn                   | demo     | USD/bushel             | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| soybeans               | demo     | USD/bushel             | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| natural_gas            | demo     | USD/mmBtu              | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| urea                   | demo     | USD/metric_ton         | 2026-08-21 00:00:00     | 2026-08-23 00:00:00   |               5 |                   10 | False   |
| cf                     | demo     | USD/share              | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| ntr                    | demo     | USD/share              | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| mos                    | demo     | USD/share              | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| dxy                    | demo     | index                  | 2026-08-28 00:00:00     | 2026-08-28 21:00:00   |               0 |                    4 | False   |
| us_2y                  | demo     | percent                | 2026-08-27 00:00:00     | 2026-08-28 00:00:00   |               0 |                    5 | False   |
| breakeven_5y           | demo     | percent                | 2026-08-27 00:00:00     | 2026-08-28 00:00:00   |               0 |                    5 | False   |
| crop_condition         | demo     | percent_good_excellent | 2026-08-21 00:00:00     | 2026-08-22 00:00:00   |               6 |                   10 | False   |
| yield_revision         | demo     | percent                | 2026-07-31 00:00:00     | 2026-08-10 00:00:00   |              18 |                   45 | False   |
| exporter_stocks_to_use | demo     | percent                | 2026-07-31 00:00:00     | 2026-08-10 00:00:00   |              18 |                   45 | False   |
| policy_stress          | demo     | score_0_100            | 2026-08-28 00:00:00     | 2026-08-28 00:00:00   |               0 |                   10 | False   |
| weather_stress         | demo     | score_0_100            | 2026-08-21 00:00:00     | 2026-08-23 00:00:00   |               5 |                   10 | False   |

## Government Policy

The policy-amplification score uses the point-in-time aligned policy_stress series. In demo mode that series is a deterministic fixture; event-weighted aggregation is used only when structured policy-event records are explicitly supplied. Current policy-amplification score: 58.5/100.

## Farmer & Fertilizer Economics

Crop/input affordability and feedstock-adjusted fertilizer margins are measured separately; product-price inflation is not treated as producer profit.

## Macro Transmission

|              |   2026-08-27 |   2026-08-28 |
|:-------------|-------------:|-------------:|
| dxy          |     95.4466  |     95.4657  |
| us_2y        |      2.3155  |      2.28603 |
| breakeven_5y |      1.58414 |      1.57999 |
| natural_gas  |      1.69672 |      1.72942 |
| urea         |    307.662   |    307.662   |

## Cross-Market Signals

| Signal            |     Value | Interpretation                                                  |
|:------------------|----------:|:----------------------------------------------------------------|
| Wheat vs DXY      |  0.77762  | positive supports physical scarcity; negative is non-confirming |
| Corn / Urea       | -1.3586   | positive means improving fertilizer affordability               |
| Nitrogen margin   | -1.04131  | positive means stronger product/feedstock economics             |
| Grain + breakeven | -2.17886  | positive means stronger inflation transmission                  |
| DXY trend         |  0.680219 | positive is a macro headwind to dollar-priced grain             |

## Strongest Evidence For the Thesis

1. Yield Revision: +25.15 score contribution.
2. Weather Stress: +19.95 score contribution.
3. Exporter Stocks To Use: +8.65 score contribution.

## Strongest Evidence Against the Thesis

1. Simple benchmark time_series_momentum achieved Sharpe 0.56 versus -0.19 for the regime strategy in this sample.
2. Corn Urea: -10.46 score contribution.

## Invalidation

1. crop conditions deteriorate, yield expectations fall, or inventories tighten
2. the return spread divergence reverses or loses statistical significance
3. the beta neutral divergence reverses or loses statistical significance

## Portfolio & Risk

| Asset   |      Weight |
|:--------|------------:|
| CF      | -0.0103902  |
| DXY     |  0.0492849  |
| MOS     | -0.0273668  |
| NG      |  0.00302481 |
| NTR     |  0.0163006  |
| US2Y    |  0          |
| ZC      | -0.0248748  |
| ZS      | -0.0299441  |
| ZW      | -0.0221563  |

|   gross_exposure |   net_exposure |   long_exposure |   short_exposure |   expected_volatility |   historical_var |   historical_cvar |   var_confidence |   historical_priced_weight_coverage_mean |   historical_priced_weight_coverage_min |   historical_unpriced_days_excluded |   historical_risk_observations | largest_risk_contributor   |   static_weight_replay_drawdown |   current_drawdown |   simulated_strategy_drawdown |   simulated_drawdown_risk_multiplier |   pre_control_gross_exposure |   post_control_gross_exposure |
|-----------------:|---------------:|----------------:|-----------------:|----------------------:|-----------------:|------------------:|-----------------:|-----------------------------------------:|----------------------------------------:|------------------------------------:|-------------------------------:|:---------------------------|--------------------------------:|-------------------:|------------------------------:|-------------------------------------:|-----------------------------:|------------------------------:|
|         0.183342 |     -0.0461219 |       0.0686102 |         0.114732 |            0.00898629 |      0.000959957 |        0.00123771 |             0.95 |                                 0.999794 |                                       0 |                                   1 |                           4849 | MOS                        |                      -0.0185759 |                nan |                     -0.170854 |                                  0.2 |                     0.916712 |                      0.183342 |

## Scenario Analysis

| scenario            |   portfolio_pnl |   gross_loss |   gross_gain | worst_contributor   | best_contributor   | description                                                                |
|:--------------------|----------------:|-------------:|-------------:|:--------------------|:-------------------|:---------------------------------------------------------------------------|
| base_case           |      0          | -0           |  0           | CF                  | CF                 | Current consensus trajectory with no discrete shock.                       |
| bullish_grain_shock |     -0.00819621 |  0.00872303  |  0.000526827 | ZW                  | NTR                | Weather deteriorates, exporter restrictions rise, and inventories tighten. |
| bearish_grain       |      0.00721245 |  0.000519265 |  0.00773172  | NTR                 | ZW                 | Rainfall and yields recover while reserves and exports normalize.          |
| stagflation         |     -0.00169925 |  0.00478877  |  0.00308951  | ZW                  | DXY                | Grain and oil remain elevated, breakevens rise, and policy turns hawkish.  |

## Benchmark Comparison

| benchmark            |   total_return |   annualized_return |   annualized_volatility |   sharpe_ratio |   sortino_ratio |   maximum_drawdown |   calmar_ratio |   win_rate |   payoff_ratio |   annualized_turnover |   average_holding_period |   number_of_trades |   tail_loss |     var_95 |    cvar_95 |   total_transaction_cost |   average_gross_exposure |   average_net_exposure |   missing_return_observations |   unpriced_exposure_observations |   unpriced_exposure_periods |
|:---------------------|---------------:|--------------------:|------------------------:|---------------:|----------------:|-------------------:|---------------:|-----------:|---------------:|----------------------:|-------------------------:|-------------------:|------------:|-----------:|-----------:|-------------------------:|-------------------------:|-----------------------:|------------------------------:|---------------------------------:|----------------------------:|
| time_series_momentum |      0.531614  |          0.0250281  |               0.0456173 |       0.564731 |        0.819935 |          -0.192465 |      0.13004   |   0.514266 |       1.04649  |              8.69521  |                  41.5388 |                837 |  0.00457023 | 0.00457023 | 0.00670311 |                0.113947  |                 0.578511 |            -0.0438489  |                             0 |                                0 |                           0 |
| equal_weight         |      0.192146  |          0.0102431  |               0.0285805 |       0.37086  |        0.549378 |          -0.185    |      0.0553683 |   0.506903 |       1.04404  |              1.59637  |                4346      |                  8 |  0.00278809 | 0.00278809 | 0.00422859 |                0.0247636 |                 0.353748 |             0.353748   |                             0 |                                0 |                           0 |
| static_long_ZW       |      0.13374   |          0.00730484 |               0.0543081 |       0.161158 |        0.233714 |          -0.23974  |      0.0304699 |   0.496318 |       1.04731  |              0.746779 |                4346      |                  1 |  0.00424593 | 0.00424593 | 0.00779832 |                0.0147273 |                 0.276006 |             0.276006   |                             0 |                                0 |                           0 |
| strategy             |     -0.0787344 |         -0.00474383 |               0.0238532 |      -0.187422 |       -0.260578 |          -0.211811 |     -0.0223965 |   0.49977  |       0.964848 |             15.2264   |                  13.6908 |               2309 |  0.00236838 | 0.00236838 | 0.00384682 |                0.187528  |                 0.323486 |            -0.00126413 |                             0 |                                0 |                           0 |

## Historical Stress Tests

| stress_period                | start               | end                 |   n_observations |   priced_observations |   portfolio_pnl |   worst_daily_pnl |   maximum_drawdown |   annualized_volatility |   mean_priced_weight_coverage |   minimum_priced_weight_coverage |   excluded_unpriced_days |
|:-----------------------------|:--------------------|:--------------------|-----------------:|----------------------:|----------------:|------------------:|-------------------:|------------------------:|------------------------------:|---------------------------------:|-------------------------:|
| 2007-08_global_food_crisis   | 2008-01-28 00:00:00 | 2008-06-30 00:00:00 |              111 |                   110 |     0.00015471  |       -0.00154791 |        -0.00804955 |              0.0092407  |                      0.990991 |                                0 |                        1 |
| 2010-11_commodity_shock      | 2010-06-01 00:00:00 | 2011-04-29 00:00:00 |              239 |                   239 |    -0.000787068 |       -0.0023013  |        -0.0172184  |              0.0100946  |                      1        |                                1 |                        0 |
| 2012_us_drought              | 2012-05-01 00:00:00 | 2012-09-28 00:00:00 |              109 |                   109 |    -0.0234961   |       -0.00160436 |        -0.0263014  |              0.00976191 |                      1        |                                1 |                        0 |
| 2020_covid_supply_disruption | 2020-02-20 00:00:00 | 2020-05-29 00:00:00 |               72 |                    72 |    -0.00651247  |       -0.00187836 |        -0.00943643 |              0.00869219 |                      1        |                                1 |                        0 |
| 2021_fertilizer_shock        | 2021-08-02 00:00:00 | 2021-12-31 00:00:00 |              110 |                   110 |     0.00574218  |       -0.00162153 |        -0.00525623 |              0.00930058 |                      1        |                                1 |                        0 |
| 2022_russia_ukraine_invasion | 2022-02-18 00:00:00 | 2022-04-29 00:00:00 |               51 |                    51 |     0.00292387  |       -0.00104072 |        -0.002158   |              0.0076984  |                      1        |                                1 |                        0 |
| 2023_black_sea_disruption    | 2023-07-10 00:00:00 | 2023-08-31 00:00:00 |               39 |                    39 |     0.00148477  |       -0.0015339  |        -0.00302957 |              0.0102324  |                      1        |                                1 |                        0 |

## Upcoming Catalysts

| date                | event                | affected_regime                 | importance   | expected_importance   | calendar_quality      |
|:--------------------|:---------------------|:--------------------------------|:-------------|:----------------------|:----------------------|
| 2026-08-31 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-09-07 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-09-10 00:00:00 | ENSO outlook update  | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-09-14 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-09-14 00:00:00 | WASDE release        | inventory_food_security_stress  | high         | high                  | deterministic_fixture |
| 2026-09-15 00:00:00 | US CPI release       | food_inflation_transmission     | high         | high                  | deterministic_fixture |
| 2026-09-21 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-09-28 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-10-05 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-10-07 00:00:00 | FOMC decision proxy  | monetary_tightening_stagflation | high         | high                  | deterministic_fixture |
| 2026-10-12 00:00:00 | Crop progress update | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-10-12 00:00:00 | ENSO outlook update  | weather_crop_supply_shock       | medium       | medium                | deterministic_fixture |
| 2026-10-12 00:00:00 | WASDE release        | inventory_food_security_stress  | high         | high                  | deterministic_fixture |

## Data Quality

Status: **warning**; score: **96.5/100**.

1. 9 changes exceed robust z=8
2. 1 changes exceed robust z=8

## Bottom Line

The system locates the cycle in Supply Response / Mean Reversion, led by yield revision. The highest-probability next state is Supply Response / Mean Reversion. Qualified ranked signals express that measurement with conservative risk controls. The view is invalidated if the leading score and its factor contributions reverse. All outputs are research candidates, not automatic orders.

## Technical Appendix

See `monitor.json` and the CSV artifacts for exact factor values, contributions, transitions, weights, risk, and reproducibility metadata.
