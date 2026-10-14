# Real-data grain prediction experiment

This experiment asks a narrower question than the regime monitor: do public,
point-in-time agricultural fundamentals and weather add out-of-sample forecast
value for 21-session wheat, corn, and soybean futures returns beyond simple
market baselines?

The frozen specification is `config/predictive_experiment.yaml`. It must not be
edited after the locked holdout is opened. A new specification ID and a new
unseen holdout are required for any subsequent tuning cycle.

## Data contract

- **Market:** Yahoo `ZW=F`, `ZC=F`, and `ZS=F` continuous-futures proxies.
  These are not individual exchange contracts and are not confirmatory-quality
  P&L data. The primary exploratory series neutralizes the return on the first
  valid session on or after the 15th in every listed delivery month; the raw
  vendor series is retained as a sensitivity. This deterministic rule avoids
  selecting repairs from return magnitude, but also removes that session's
  genuine market move. A confirmatory rerun requires individual contracts, a
  causal roll rule, official settlements, and explicit two-leg roll P&L.
- **USDA:** monthly historical WASDE CSVs from April 2010 onward. Values are
  keyed by report, commodity, region, market year, and attribute before a
  revision is calculated. Because the consolidated historical CSV is posted
  the day after the report, the CSV-only experiment activates each observation
  one calendar day after `ReleaseDate` and trades no earlier than the following
  market session.
- **Weather:** NOAA NCEI Daily Summaries for fixed preregistered crop-region
  station baskets. Observations are delayed by three calendar days. NOAA's
  current historical files are quality-controlled endpoint files, not archived
  daily vintages, so weather remains an operational PIT proxy rather than a
  strict vintage source.

Raw and processed inputs, file hashes, date coverage, and acquisition failures
are written to the experiment data-lineage artifact.

## Target and sampling

Decisions use the final observed market row of each `W-FRI` period. Features are
known at that close. The target starts at the next market-session close and
ends 21 sessions later; regression labels are gross proxy returns. Trading
costs are charged only in the path-dependent portfolio backtest.

Wheat crop years begin June 1. Corn and soybean crop years begin September 1.
Every outer test fold is one complete asset-specific crop year. Training is
expanding, must contain at least five prior crop years, and excludes labels
whose target interval reaches the test boundary or its five-business-day
embargo.

The latest three complete crop years for each asset are locked holdout folds.
Candidate feature families are selected using development folds only. The
global selection cutoff precedes the earliest holdout start across all three
assets, so a later wheat development year cannot overlap a corn/soybean
holdout. The primary development-selected family is then locked for both the
repaired and raw-price holdouts alongside the preregistered simple baselines.

## Models

The fixed baselines are:

1. fold historical mean;
2. training-only crop-seasonality mean;
3. raw 60-session momentum;
4. ridge regression on causal price/calendar features.

The ablations add USDA, weather, and USDA plus weather to the same ridge core.
Ridge alpha is selected by an inner expanding split using the last prior crop
year. Winsorization, imputation, missingness indicators, standardization, and
constant-column removal are fitted on training rows only.

## Evaluation and claim policy

The primary predictive endpoint is pooled out-of-sample R-squared relative to
the fold historical-mean forecast. Secondary diagnostics include RMSE, MAE,
Pearson and Spearman information coefficients, directional accuracy, and
asset/crop-year attribution.

All forecast models use the same weekly position translation, one-session
execution lag, exposure limits, and cost assumptions. Portfolio evidence uses
the common intersection of the three asset holdout calendars. Results are
repeated at 1.0x, 1.5x, and 2.0x costs; the explicit roll charge is applied to
absolute held exposure on each preregistered delivery-month anchor session.

The current experiment is a **provisional falsification test**, not proof of a
tradable strategy. It cannot be promoted to confirmatory evidence while it
uses vendor continuous proxies or endpoint weather files. A positive result
must also survive individual-contract data, a newly untouched holdout,
dependence-aware confidence intervals, cost stress, and asset/year
concentration checks.

## Running

After installing the package, the integrated command is:

```bash
agri-macro predictive-experiment \
  --experiment-config config/predictive_experiment.yaml \
  --data-dir data/predictive_experiment_v1 \
  --output-dir reports/predictive_experiment_v1
```

Use `--refresh-data` only before opening a new, separately identified
experiment. Cached raw files are reused by default.
