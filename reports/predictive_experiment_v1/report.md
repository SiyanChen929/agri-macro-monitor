# Predictive experiment: zw_zc_zs_21d_v1

**Outcome: `exploratory_support_only`.** The predeclared checks are directionally positive, but this is not a validated predictive edge: vendor continuous proxies and non-vintage weather still require an untouched contract-level confirmation.

This was one execution of the frozen exploratory specification. Candidate families were compared only on expanding development folds; each predeclared proxy treatment then opened its latest three complete crop-year holdouts once. No model or alpha was changed after those results were observed.

## What ran

- As-of date: 2026-08-28
- Assets: ZW, ZC, ZS
- Target: next-session through 21-session return
- Primary selected family: `ridge_core`
- Family locked into raw-stitch sensitivity: `ridge_core`
- Raw-only development preference (diagnostic, not used): `ridge_core`
- Decisions: final priced session in each W-FRI period
- Validation: expanding crop-year folds, label-overlap purge, five-business-day embargo
- Portfolio window: common intersection of all three asset holdouts

## Primary delivery-anchor-neutralized proxy metrics

| split       | model                   |   observations |   oos_r2_vs_historical_mean |   rmse |   spearman_ic |   directional_accuracy |
|:------------|:------------------------|---------------:|----------------------------:|-------:|--------------:|-----------------------:|
| development | historical_mean         |           1147 |                      0      | 0.0712 |       -0.0155 |                 0.4821 |
| development | momentum_60             |           1147 |                     -2.6136 | 0.1353 |       -0.0472 |                 0.4847 |
| development | ridge_core              |           1147 |                     -0.0283 | 0.0722 |       -0.0332 |                 0.4821 |
| development | ridge_core_usda         |           1147 |                     -0.0206 | 0.0719 |       -0.0301 |                 0.4778 |
| development | ridge_core_usda_weather |           1147 |                     -0.0441 | 0.0728 |       -0.0264 |                 0.4752 |
| development | ridge_core_weather      |           1147 |                     -0.0323 | 0.0723 |       -0.012  |                 0.4917 |
| development | seasonality             |           1147 |                     -0.0745 | 0.0738 |       -0.0357 |                 0.4874 |
| holdout     | historical_mean         |            471 |                      0      | 0.0585 |       -0.011  |                 0.482  |
| holdout     | momentum_60             |            471 |                     -2.8094 | 0.1141 |       -0.2033 |                 0.5032 |
| holdout     | ridge_core              |            471 |                      0.0149 | 0.058  |        0.1059 |                 0.4926 |
| holdout     | seasonality             |            471 |                      0.0011 | 0.0584 |        0.0598 |                 0.4968 |

## Raw vendor-stitch sensitivity

| split       | model                   |   observations |   oos_r2_vs_historical_mean |   rmse |   spearman_ic |   directional_accuracy |
|:------------|:------------------------|---------------:|----------------------------:|-------:|--------------:|-----------------------:|
| development | historical_mean         |           1147 |                      0      | 0.0746 |       -0.0614 |                 0.5397 |
| development | momentum_60             |           1147 |                     -2.734  | 0.1442 |       -0.0819 |                 0.4717 |
| development | ridge_core              |           1147 |                     -0.0142 | 0.0752 |        0.0175 |                 0.5074 |
| development | ridge_core_usda         |           1147 |                     -0.0216 | 0.0754 |       -0.0004 |                 0.51   |
| development | ridge_core_usda_weather |           1147 |                     -0.0293 | 0.0757 |        0.0012 |                 0.5013 |
| development | ridge_core_weather      |           1147 |                     -0.0298 | 0.0757 |        0.0233 |                 0.4961 |
| development | seasonality             |           1147 |                     -0.0664 | 0.0771 |        0.0091 |                 0.5065 |
| holdout     | historical_mean         |            471 |                      0      | 0.0632 |       -0.1335 |                 0.4289 |
| holdout     | momentum_60             |            471 |                     -2.8154 | 0.1235 |       -0.2501 |                 0.4437 |
| holdout     | ridge_core              |            471 |                      0.0259 | 0.0624 |        0.1755 |                 0.5478 |
| holdout     | seasonality             |            471 |                      0.0572 | 0.0614 |        0.1889 |                 0.5541 |

## Selected-forecast implementation path

| variant                             | evaluation_split   | common_first_decision   | common_last_decision   | selected_model   |   cost_multiplier |   observations |   annualized_net_return |   annualized_net_volatility |   net_sharpe_zero_rate |   maximum_drawdown |   average_gross_exposure |   annualized_turnover |   total_transaction_cost |   total_roll_cost |   terminal_net_nav |
|:------------------------------------|:-------------------|:------------------------|:-----------------------|:-----------------|------------------:|---------------:|------------------------:|----------------------------:|-----------------------:|-------------------:|-------------------------:|----------------------:|-------------------------:|------------------:|-------------------:|
| delivery_anchor_neutralized_primary | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               1   |            569 |                  0.0467 |                      0.0658 |                 0.7092 |            -0.1009 |                   0.3977 |                9.2942 |                   0.0147 |            0.0015 |             1.1085 |
| delivery_anchor_neutralized_primary | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               1.5 |            569 |                  0.0429 |                      0.0658 |                 0.6521 |            -0.1024 |                   0.3977 |                9.2942 |                   0.022  |            0.0023 |             1.0995 |
| delivery_anchor_neutralized_primary | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               2   |            569 |                  0.0392 |                      0.0658 |                 0.5952 |            -0.1038 |                   0.3977 |                9.2942 |                   0.0294 |            0.003  |             1.0907 |
| raw_vendor_stitch_sensitivity       | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               1   |            569 |                  0.0877 |                      0.0745 |                 1.1764 |            -0.0975 |                   0.4476 |                9.2028 |                   0.0145 |            0.0015 |             1.2089 |
| raw_vendor_stitch_sensitivity       | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               1.5 |            569 |                  0.0838 |                      0.0745 |                 1.1251 |            -0.0992 |                   0.4476 |                9.2028 |                   0.0218 |            0.0023 |             1.1993 |
| raw_vendor_stitch_sensitivity       | holdout            | 2023-06-02 00:00:00     | 2025-08-29 00:00:00    | ridge_core       |               2   |            569 |                  0.0799 |                      0.0745 |                 1.0738 |            -0.1008 |                   0.4476 |                9.2028 |                   0.0291 |            0.0031 |             1.1896 |

Signals use `tanh(forecast / training-target-volatility)`, a one-third asset cap, and a 1.0 gross cap. With close-only data, the next-session close is treated as execution and P&L begins on the following close-to-close interval. Transaction and slippage costs are charged on position changes. The 3 bps roll charge is applied to absolute held exposure on each preregistered delivery-month anchor session; it is still a proxy, not observed two-leg roll P&L.

## Point-in-time controls

- Historical USDA CSV values become eligible one calendar day after report date and activate only on the first asset market session strictly after that date.
- USDA revisions are event features active only through the activation week's decision. The latest stocks-to-use level is carried causally with days-since-release.
- NOAA observations are reconstructed with a three-calendar-day lag and joined backward; no market decision can use a later weather observation.
- Winsorization, missing-value handling, standardization, alpha choice, and label purging use training data only.

## Data lineage

| dataset       | provider                                                   | quality                                                 |   rows | first_date          | last_date           | failures                                                                                                                         |
|:--------------|:-----------------------------------------------------------|:--------------------------------------------------------|-------:|:--------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| market_prices | Yahoo chart                                                | vendor_continuous_proxy_not_contract_level              |   4439 | 2009-01-02T00:00:00 | 2026-08-27T00:00:00 |                                                                                                                                  |
| weather       | NOAA NCEI Daily Summaries                                  | official_observations_conservative_3d_lag               |  19335 | 2009-01-01T00:00:00 | 2026-08-24T00:00:00 |                                                                                                                                  |
| usda_wasde    | USDA WAOB historical WASDE; pinned archive replay fallback | official_as_published_csvs_validated_headers_and_hashes |    582 | 2010-04-09T00:00:00 | 2026-08-12T00:00:00 | 2013-10: no release found in acquired official files; 2019-01: no release found in acquired official files; 2025-10: unavailable |

## Limits and claim boundary

This is a falsification test, not a deployable trading model. Yahoo's generic series are vendor-stitched rather than individual contracts; the NOAA endpoint files are current quality-controlled histories rather than archived daily vintages; and overlapping 21-session labels reduce the effective independent sample size. A positive exploratory result still requires individual-contract settlements, a causal roll implementation, dependence-aware uncertainty, and a newly untouched holdout before any predictive or tradability claim.
