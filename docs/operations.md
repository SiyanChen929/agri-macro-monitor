# Operations

This guide covers installation, CLI use, configuration, run artifacts, scheduling, and the path from the completed research foundation to a production-quality deployment. Model equations and assumptions are in [Methodology](methodology.md).

## Installation

Requirements:

- Python 3.11 or newer;
- a filesystem location writable for the virtual environment and reports;
- network access only for `live` mode or package installation;
- `FRED_API_KEY` only for live FRED/ALFRED series.

Create an ordinary local virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This procedure is intentionally independent of `/opt`, Conda, Homebrew, or any machine-specific Python installation. On Windows PowerShell, activation is `.venv\Scripts\Activate.ps1`; the remaining `python -m pip` commands are unchanged.

Install test support with:

```bash
python -m pip install -e '.[dev]'
```

Runtime dependencies are `numpy`, `pandas`, `matplotlib`, `PyYAML`, and `tabulate`. `requirements.txt` is available for non-editable environments, but an editable package install is preferred because it creates the `agri-macro` entry point. Dependency ranges are constrained but not locked; archive `python -m pip freeze` alongside governed research if byte-for-byte environment recreation matters.

Verify the install:

```bash
agri-macro --help
agri-macro show-config
python -m pytest
```

`python -m agri_macro_quant` is equivalent to `agri-macro` after installation.

## CLI commands

### Show resolved configuration

```bash
agri-macro show-config
agri-macro show-config --config /absolute/path/to/project/config/config.yaml
```

The JSON summary includes project root, default mode/start/calendar, canonical series, and reports directory.

### Validate ingestion and alignment only

```bash
agri-macro validate-data \
  --mode demo \
  --start 2023-01-01 \
  --as-of 2025-12-31
```

Options are `--config`, `--mode {demo,live,cache}`, `--start`, `--as-of`, and `--cache-path`. The command prints the complete quality report. Exit status is `0` for healthy or warning and `2` for failed validation or an operational/configuration error.

### Run the complete workflow

```bash
agri-macro run \
  --mode demo \
  --start 2023-01-01 \
  --as-of 2025-12-31
```

Run options:

| Option | Behavior |
|---|---|
| `--config PATH` | Load a different main YAML bundle |
| `--mode demo|live|cache` | Override `runtime.mode` |
| `--start YYYY-MM-DD` | Override the first reported/backtested evaluation date; causal warm-up data are fetched before it |
| `--as-of YYYY-MM-DD` | Cap the information set and final decision date |
| `--cache-path PATH` | Select the canonical snapshot used in cache mode |
| `--output-dir PATH` | Use a named directory instead of a unique UTC-stamped run directory |
| `--overwrite` | Completely replace a non-empty named directory only when it contains a valid prior agri-macro run marker/manifest; unmarked directories are refused |
| `--fail-on-quality` | Write `quality_failure.json`, stop, and return exit status `2` when quality fails |
| `--skip-statistical` | Omit the surrogate-label statistical regime diagnostic; transparent regimes and transitions still run |
| `--no-research-log` | Do not append this run to the project-level JSONL research log |

Without `--output-dir`, output is atomically allocated under a collision-resistant `reports/run_<UTC timestamp>_<token>/` name. A non-empty named directory is rejected unless `--overwrite` is present, and overwrite is restricted to a directory previously marked by this application. This prevents both same-second run collisions and accidental recursive deletion of an unrelated directory.

The pipeline fetches and aligns `runtime.warmup_business_days` (504 by default) before `--start`. Features, covariance estimates, and walk-forward models are formed causally on that warm-up history; positions, P&L, costs, benchmarks, metrics, and report performance begin exactly at `--start`. The manifest records both `fetch_start` and `evaluation_start`.

The success payload's `signals` field is the number of full-history signal records, not the number of current recommendations. Current candidates are the latest-date rows in `data/processed/signals.csv.gz` and the ranked subset in the report.

## Operating modes

### Demo

```bash
agri-macro run --mode demo --start 2018-01-01 --as-of 2025-12-31
```

Demo mode is deterministic for the same date range, configuration, code, and `runtime.random_seed`. It is network-free and marks observations/reports/signals as synthetic. Use it for installation checks, regression tests, demonstrations, and report development—not performance claims.

### Cache

Each completed run saves a portable canonical snapshot and adjacent checksum manifest. Reuse it directly:

```bash
agri-macro run \
  --mode cache \
  --cache-path /absolute/path/to/run/data/raw/observations.csv.gz \
  --start 2020-01-01 \
  --as-of 2025-12-31 \
  --fail-on-quality
```

The companion file must be named `observations.csv.gz.manifest.json`. Cache loading verifies SHA-256 by default and raises on a missing or mismatched manifest. If `--cache-path` is omitted, the default is `data/cache/latest.csv.gz` with `data/cache/latest.csv.gz.manifest.json`.

Canonical cache rows require at least `series_id`, `observation_date`, `publication_date`, and `value`. For controlled point-in-time research, also supply `effective_date`, `revision_date`, `source`, `frequency`, `unit`, and `vintage_id`. Missing effective/revision dates are conservatively defaulted to observation/publication dates, but that is not a substitute for genuine vintage history.

### Live

The shipped live ingestor supports:

- Yahoo chart data for configured wheat, corn, soybean, natural-gas, CF, NTR, MOS, and DXY proxies;
- FRED/ALFRED for the US 2-year rate and 5-year breakeven.
- local or HTTP CSV sources configured with `source: local_csv`, a `location`/`path`/`source_id`, and optional `date_column`/`value_column` fields.

Set the FRED credential in the process environment or scheduler's protected secret environment:

```bash
export FRED_API_KEY='replace-with-your-key'
agri-macro validate-data --mode live --start 2020-01-01 --as-of 2025-12-31
```

The default `data_sources.yaml` still marks urea, crop condition, yield revision, exporter stocks-to-use, policy stress, and weather stress as demo sources. Live mode deliberately omits them and records `live_source_not_configured`; it never mixes synthetic values into a live run. Missing required series make quality fail.

Use this operational gate so incomplete feeds stop the job transparently:

```bash
agri-macro run \
  --mode live \
  --start 2020-01-01 \
  --fail-on-quality
```

Without `--fail-on-quality`, the pipeline still refuses to publish positions if the latest date cannot produce a regime score; it writes `quality_failure.json`. Other failed-quality cases set research confidence to zero. Scheduled live work should always use `--fail-on-quality`.

Do not treat live mode as production-ready until every critical demo-only series has a real, licensed or authoritative, revision-aware adapter and the end-to-end validation passes. Endpoint failure, missing credentials, empty responses, and unsupported sources are surfaced as ingestion/quality issues rather than silently synthesized.

## Configuration

The default bundle is rooted at `config/config.yaml`. A custom main file is expected at `<custom-project>/config/config.yaml`; its project root is the parent of that `config/` directory. Included component paths and data/report paths are resolved against that root, not the caller's working directory. Environment variables and `~` are expanded in configured filesystem paths.

Every run stores the resolved semantic research configuration and effective evaluation/fetch bounds in `configuration_snapshot.json`. `manifest.json` records a stable semantic configuration hash separately from invocation metadata such as the output directory and logging gates.

### `config/config.yaml`

Active pipeline settings include:

- `runtime.mode`, evaluation `start_date`, `end_date`, business calendar, UTC decision time, random seed, and causal warm-up length;
- report and default cache paths; `data_dir`, `raw_dir`, and `processed_dir`
  are resolved and recorded for provenance, while each run writes its raw and
  processed artifacts beneath that run's output directory;
- feature change/z-score windows and winsor quantiles;
- execution lag, rebalance frequency, annualization, starting capital, transaction/slippage/roll costs;
- target volatility, position/gross/asset-class limits, and drawdown schedule.

`runtime.strict_point_in_time` is enforced: setting it false is rejected because the CLI has no unsafe bypass. `features.minimum_periods_fraction` and `features.include_data_age`, the factor-normalization block, model refit settings, and the quality warning/failure/coverage/jump/penalty settings are wired into the run. Age features are also saved separately for audit.

### `config/data_sources.yaml`

Each series may configure:

- source and source identifier;
- daily/weekly/monthly/etc. frequency and unit;
- publication-lag approximation and maximum staleness;
- criticality;
- min/max values, maximum absolute daily return, and price scaling.

The `sources` section provides live endpoint URLs, timeouts, and the FRED API-key environment variable. Its provider blocks are schema-checked and must declare the matching adapter (`demo`, `yahoo_chart`, `fred`, or `csv`); unknown providers, misspelled settings, adapter mismatches, and series references to unconfigured providers fail at load time. The live dispatcher registers Yahoo, FRED, and `local_csv`/`csv`. A simple CSV may provide configured date/value columns and receive the declared publication lag; a canonical long CSV may preserve genuine publication, effective, revision, and vintage timestamps. The latter is preferred for governed backtests.

### `config/regime_thresholds.yaml`

The transparent scorer actively reads `weights`, `intercepts`, optional `labels`, `temperature`, `min_coverage`, `rescale_missing`, and `raw_clip`. Canonical factor/regime names and the documented aliases are normalized; unknown top-level fields, regimes, factors, intercept/label targets, or nested definition fields raise an error.

Transition horizons and walk-forward refit controls live under `models.transitions` in `config.yaml`; the shipped values are 5/21/63 business-day rows. Statistical-regime refit controls are under `models.statistical_regime`. Only keys actually consumed by the transparent scorer remain in `regime_thresholds.yaml`.

### `config/assets.yaml`

This file is an instrument/reference catalog and is captured for provenance. The v0.1 pipeline's traded-series mapping, default signal specifications, relative-value pairs, and factor exposures are code-defined. Changing only `assets.yaml` does not add an ingested or traded asset, and contract multipliers do not affect backtest P&L.

### `config/catalysts.yaml`

The CLI expands `monthly`, `weekly`, and fixed-interval recurrence rules over a 45-day horizon and applies a weekday-only “following” adjustment. The shipped dates are explicitly labelled `deterministic_fixture`.

Changing the label does not make dates official. Production deployment needs an authoritative release-calendar ingestion path and verification for USDA, NOAA, BLS, and Federal Reserve dates. Until then, reports must describe catalysts as fixtures, including in live research.

## Data quality and failure policy

Quality statuses are:

- `healthy`: score at or above 85 and no warning, error, or critical issue;
- `warning`: at least one warning but no error/critical issue, or a score below 85 but at least 60;
- `failed`: score below 60 or any error/critical issue.

Warnings reduce current-date candidate confidence by `quality_score / 100`; failures set current candidate confidence to zero. Historical backtest confidence instead uses a causal, date-indexed usable-critical-series coverage multiplier, so a later endpoint warning or failure cannot rewrite earlier decisions. The signal artifact retains `raw_confidence`, `backtest_confidence`, and `current_quality_multiplier` for audit. Error/critical examples include duplicate or nonfinite vintages, impossible bounds, missing required columns, revision before publication, missing critical series, insufficient critical-series coverage, look-ahead, and stale values leaking through alignment. Invalid aligned values are quarantined before feature construction and recorded in `diagnostics/quarantined_value_mask.csv.gz`.

Recommended gates:

```bash
agri-macro validate-data --mode cache --cache-path /path/to/latest.csv.gz
agri-macro run --mode cache --cache-path /path/to/latest.csv.gz --fail-on-quality
```

Review `data_quality.json`, `diagnostics/series_metadata.csv`, `diagnostics/factor_coverage.csv`, and `diagnostics/model_health.json` on every governed run. A quality pass does not certify economic correctness, licensing, or investability; it certifies only the implemented checks.

## Run artifacts

### Top level

| Artifact | Purpose |
|---|---|
| `report.html` | Standalone executive HTML page with relative links to local chart images |
| `report.md` | Readable text report |
| `technical_report.md` | Audit-oriented scores, contributions, transition vector, signals, weights, risks, metrics, benchmarks, scenarios, stresses, and quality summary |
| `monitor.json` | Machine-readable current monitor payload |
| `manifest.json` | Version, git revision or unversioned marker, config/data hashes, timestamp, mode, seed, platform, and output listing |
| `configuration_snapshot.json` | Main and included YAML plus resolved paths |
| `data_quality.json` | Full issue list, per-series details, status, score, and confidence multiplier |
| `quality_failure.json` | Written only when `--fail-on-quality` stops a failed run |

### `data/raw/`

- `observations.csv.gz`: canonical long-form vintages;
- `observations.csv.gz.manifest.json`: snapshot checksum and coverage metadata.

### `data/processed/`

- `market_point_in_time.csv.gz`, `availability_timestamps.csv.gz`, `data_age_days.csv.gz`;
- `features.csv.gz`, `mvp_factors.csv`, `economic_features.csv.gz`;
- `regime_scores.csv`, `signals.csv.gz`, `current_portfolio.csv`;
- `backtest_equity.csv`, `backtest_returns.csv`, `backtest_positions.csv.gz`, `backtest_costs.csv`, `backtest_metrics.json`;
- `upcoming_catalysts.csv`.

### `diagnostics/`

- raw regime scores, factor contributions, regime/factor coverage, and series metadata;
- optional statistical regime probabilities, coefficients, and metadata;
- transition matrices, causal forecast histories, and transition metadata;
- cost-matched benchmark comparison, forward scenarios, historical stress tests;
- lead/lag correlations, distributed-lag regressions, Granger tests, metadata, and the ten-question tested/blocked audit;
- the pre-stop/pre-drawdown current portfolio for control attribution;
- model health and factor drift.

### `charts/`

When data are available, the report creates regime history, wheat/DXY, corn/urea, exporter stocks-to-use, grain/rates/breakeven, fertilizer margin, equity curve, drawdown, signal heatmap, and one-month transition charts. Missing inputs cause the affected chart to be skipped rather than fabricated.

### Project research log

Unless disabled, `reports/research_log.jsonl` receives one append-only record per run with hashes, mode, dominant regime, and signal/portfolio artifact paths. `realized_outcome_status` is initially `pending`; v0.1 does not include an automatic outcome updater. Archive or rotate this file under an explicit retention policy, and avoid overlapping writers because no cross-process file lock is implemented.

## Reproducibility and retention

For a reproducible research record, retain together:

1. the complete run directory;
2. the exact Git revision when available, or the source archive matching `source_tree_sha256` when the run was outside Git;
3. the Python environment lock/freeze;
4. source-data licensing and extraction metadata;
5. the research-log record and later realized-outcome annotation.

The manifest separately hashes the semantic research configuration, operational invocation, source tree, raw observation/vintage frame, serialized raw snapshot and its manifest, aligned value panel, and aligned availability panel. It also records installed direct-dependency versions and upstream cache lineage. A fixed seed makes the demo and model initialization deterministic, but platform/library-version differences can still change floating-point results or rendered charts.

## Scheduling example

Use the virtual environment's executable directly; a scheduler should not depend on interactive shell activation. Create the log directory once:

```bash
mkdir -p /absolute/path/to/agri_macro_quant/reports/logs
```

This crontab example runs a verified cache after midnight Tuesday through Saturday in the cron daemon's local timezone, producing a unique run directory each time:

```cron
15 0 * * 2-6 cd /absolute/path/to/agri_macro_quant && /absolute/path/to/agri_macro_quant/.venv/bin/agri-macro run --mode cache --cache-path /absolute/path/to/verified/latest.csv.gz --fail-on-quality >> /absolute/path/to/agri_macro_quant/reports/logs/scheduler.log 2>&1
```

Before scheduling:

- validate the exact command manually under the scheduler's user;
- make the cache and its manifest readable but not broadly writable;
- use a secret manager or protected service environment for live credentials—never put a key in crontab or the command line;
- confirm timezone and data-release timing;
- ensure jobs cannot overlap;
- alert on non-zero exit, missing fresh run directory, failed quality, model-health warning, and stale critical series;
- back up or expire reports under a documented retention policy.

After all critical live adapters and official catalysts are implemented, `--mode live` may replace `--mode cache`, still with `--fail-on-quality`. The system itself does not send alerts; the scheduler or external monitoring service must inspect exit status and artifacts.

## Phased extension roadmap

The current repository is the executable research foundation. The following phases are extensions, not claims about existing behavior.

### Phase 1 — Production data completeness

- Add authoritative, licensed, revision-aware feeds for crop condition, weather, yield/production revisions, exporter stocks/use, fertilizer prices, and structured policy events.
- Wire and test local/remote canonical adapters without demo fallback.
- Replace recurrence fixtures with verified agency calendars and holiday handling.
- Add source-level retry/rate-limit policies, licensing metadata, and immutable raw landing storage.
- Define freshness service levels and require all critical feeds before publishing a live report.

### Phase 2 — Historical research and validation

- Integrate the existing event-study engine into CLI artifacts using verified event histories.
- Extend the shipped lead/lag and distributed-lag checks with local projections, regime conditioning, multiple-testing control, and authoritative event histories without equating predictive association with causality.
- Calibrate transparent weights only inside nested expanding/rolling validation, with sensitivity, bootstrap, multiple-testing, and economic-significance checks.
- Report results by historical regime and compare with momentum, carry, static, and equal-weight baselines.
- Replace public continuous proxies with governed contract histories, documented rolls, corporate actions, and asset-specific cost/liquidity models.

### Phase 3 — Independent regime discovery and calibration

- Develop externally labelled and/or unsupervised HMM, Markov-switching, mixture, clustering, Bayesian, and regularized classifier candidates.
- Separate train, calibration, model-selection, and untouched evaluation periods.
- Test persistence, probability calibration, stability, turnover, transition usefulness, and economic performance—not in-sample accuracy alone.
- Retain the transparent model as an explainable benchmark and publish disagreement diagnostics.

### Phase 4 — Portfolio and execution research

- Evaluate risk-parity/optimizer alternatives, expected-return uncertainty, covariance shrinkage, and robust factor-risk budgets.
- Add contract selection, curve/carry signals, margin, financing, borrow, dividend, FX conversion, liquidity, capacity, and market-impact models.
- Validate signal-level and regime-transition stops before wiring them into the default pipeline.
- Build instrument-level order proposals only behind independent risk checks and human approval; keep research and execution permissions separated.

### Phase 5 — Automation, governance, and monitoring

- Add an orchestrator with locking, retries, idempotency, secrets, alerts, lineage, and immutable artifact publication.
- Automate realized-outcome updates and retrospective attribution in the research log.
- Monitor source drift, feature drift, probability calibration, benchmark-relative decay, turnover, latency, and live-versus-backtest performance.
- Add model approvals, version promotion/rollback, audit trails, disaster recovery, and periodic independent validation.
- Only after legal, data-license, security, operational-risk, and investment governance approval should a separately controlled execution integration be considered.
