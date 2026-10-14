# Methodology and assumptions

This document describes the model that the CLI actually runs in v0.1. Configuration and operating procedures are covered in [Operations](operations.md).

## Research boundary

The system enforces the sequence:

```text
observations → point-in-time alignment → measurement → regimes/transitions
             → signals → constrained research portfolio → backtest/report
```

Narrative is generated only after quantitative measurement. Reports include supporting evidence, competing regimes, invalidation conditions, and benchmark counterevidence. The system estimates research exposures; it does not forecast a guaranteed return or execute a security transaction.

## Canonical observations and point-in-time alignment

Every observation vintage has this minimum identity:

```text
series_id
observation_date
publication_date
effective_date
revision_date
value
source
frequency
unit
vintage_id
```

All timestamps are normalized to timezone-naive UTC-equivalent values. A vintage is usable only at

\[
a_v = \max(\text{publication}_v,\text{revision}_v,\text{effective}_v).
\]

For each configured business-day decision timestamp \(d_t\), the aligner replays only releases with \(a_v \le d_t\). It retains the newest known vintage for each observation period, then selects the latest observation period known at that time. A later revision to an old period therefore cannot displace a newer observation or reset the newer observation's age.

The default decision time is `23:59:59` UTC. `--as-of` caps both ingestion and the decision calendar. The aligner emits four matching panels:

- the selected value;
- the selected vintage's availability timestamp;
- its economic observation date;
- calendar days since availability.

If age exceeds the series' configured `max_staleness_days`, the value is masked to missing. Defaults are 4 days for daily, 10 for weekly, 45 for monthly, 120 for quarterly, 400 for annual, and 30 for irregular series unless a series overrides them.

The data-quality layer checks schema and timestamps, duplicate vintages, revision ordering, configured bounds, return jumps, timestamp quality, missing critical series, critical coverage, future availability, aligned coverage, and stale-value leakage. A failed report has confidence multiplier zero. A warning remains usable but applies `score / 100` as the final signal-confidence multiplier.

### Publication-lag behavior by source

- Demo observations are emitted with the frequency, publication lag, staleness, and deterministic revision behavior encoded by the fixture and series specification.
- FRED requests revision vintages through its realtime/ALFRED fields. When true vintages are present, their realtime dates determine availability.
- Yahoo observations use the vendor chart timestamp as an exchange-timestamp proxy. Continuous futures are explicitly tagged as unvalidated vendor roll proxies.
- The live CSV adapter can apply `publication_lag_days` to a configured simple date/value series, or preserve the explicit timestamps in a canonical long CSV. Genuine revision-aware timestamps are required for governed PIT backtests; a configured lag is only an approximation.

`publication_lag_days` is not an additional blanket shift after alignment. It helps an adapter construct availability metadata when an authoritative release timestamp is absent. The aligner always uses the timestamps present on each vintage.

## Feature engineering

All rolling windows are trailing and never centered. The aligned value at end of day \(t\) may enter a signal dated \(t\); portfolio returns cannot be earned until the configured execution lag, one business-day row by default.

The general feature frame preserves original aligned series and adds, where configured:

- one-period log returns for positive series and simple returns otherwise;
- 5-, 20-, 60-, and 120-row changes;
- robust 60- and 252-row z-scores;
- 252-row percentile ranks;
- 20- and 60-row momentum;
- 20-row annualized realized volatility;
- 20- and 60-row moving-average distance and regression slope.

For a series \(x\), the robust normalization used by the MVP factors is

\[
z_t = \frac{\widetilde{x}_t - \operatorname{median}(x_{t-W+1:t})}
{1.4826\,\operatorname{MAD}(x_{t-W+1:t})},
\]

where \(\widetilde{x}_t\) is causally winsorized at trailing 1%/99% quantiles. If MAD is zero, the scale falls back to `IQR / 1.349`; constant histories map an unchanged observation to zero. The pipeline uses \(W=252\), at least 20 observations, and clips MVP factors to \([-4,4]\).

The generated `data_age_days.csv.gz` is an explicit audit panel. In v0.1, age is used for stale masking and reporting but is not one of the ten regime-scoring factors.

## Ten minimum-viable factors

| Factor | Raw concept | Construction in the pipeline |
|---|---|---|
| F1 `crop_condition` | Crop condition | Configured aligned level, robust z-score |
| F2 `yield_revision` | Yield-estimate revision | Configured aligned level, robust z-score |
| F3 `exporter_stocks_to_use` | Major-exporter inventory tightness | Configured aligned level, robust z-score |
| F4 `policy_stress` | Government/trade intervention stress | Configured aligned level, robust z-score |
| F5 `weather_stress` | Crop-calendar weather stress | Configured aligned level, robust z-score |
| F6 `corn_urea` | Fertilizer affordability | `corn / urea`, then robust z-score |
| F7 `dxy` | USD pressure | DXY level, robust z-score |
| F8 `us2y` | Front-end policy pressure | US 2-year yield level, robust z-score |
| F9 `breakeven` | Inflation expectations | 5-year breakeven level, robust z-score |
| F10 `wheat_momentum` | Grain price momentum | 60-row wheat log return when prices are positive, then robust z-score |

Factor signs remain economically literal. For example, high crop condition and high exporter stocks produce positive factor values; negative regime weights make them reduce supply-stress scores.

Additional signal features include crop/fertilizer affordability ratios, a farmer-margin proxy, 20-row standardized market momentum, food-inflation and grain-momentum composites, and a nitrogen producer-margin proxy:

\[
\text{NitrogenMargin}_t = \text{Urea}_t - \widehat{\beta}_{t-1}\text{NaturalGas}_t.
\]

The beta is estimated causally from 252-row changes, shifted one row, clipped to `[0, 100]`, and falls back to `2.0` when history is insufficient. These are standardized economic proxies, not company earnings forecasts.

Reusable library functions also calculate exporter surplus from major-exporter production, stocks, imports, domestic use, and required ending stocks, and aggregate policy events with trade-share weighting and exponential decay. The default CLI consumes the already configured `exporter_stocks_to_use` and `policy_stress` series rather than building them from country/event tables.

## Seven transparent regimes

The decision baseline scores:

1. Weather / Crop Supply Shock
2. Inventory / Food Security Stress
3. Government / Trade Policy Amplification
4. Agricultural Input Expansion
5. Food Inflation Transmission
6. Monetary Tightening / Stagflation
7. Supply Response / Mean Reversion

For regime \(r\), configured weights \(w_{rk}\), and available standardized factors \(z_{kt}\), weighted coverage is

\[
q_{rt} = \frac{\sum_{k\in A_t}|w_{rk}|}{\sum_k |w_{rk}|}.
\]

With the default missing-data rescaling, the raw score is

\[
u_{rt} = b_r +
\frac{\sum_k |w_{rk}|}{\sum_{k\in A_t}|w_{rk}|}
\sum_{k\in A_t} w_{rk}z_{kt}.
\]

Scores are suppressed when \(q_{rt}<0.35\), raw values are clipped to \([-12,12]\), and the displayed score is

\[
S_{rt}=100\,\sigma(u_{rt}/T),\qquad
\sigma(x)=\frac{1}{1+e^{-x}},
\]

with default temperature \(T=1\). The seven scores are independent stress/activation scores and do **not** sum to 100. The dominant regime is simply the available maximum.

`regime_contributions.csv.gz` decomposes every score by date, regime, and factor. After the 50-point neutral baseline, `score_point_contribution` values reproduce the displayed score, including adjustments for missing-factor rescaling and raw clipping. The current economic weights are in `config/regime_thresholds.yaml` and the exact resolved copy is stored with every run.

## Statistical regime diagnostic

Unless `--skip-statistical` is supplied, the pipeline also fits a dependency-free L2 multinomial logistic model in expanding walk-forward form:

- training labels are the transparent model's historical dominant regime;
- a prediction at \(t\) uses labelled observations strictly before \(t\);
- the pipeline requires at least 252 labelled rows, at least 10 examples for an eligible class, and refits every 63 rows;
- missing features are filled with training-window means and standardized using training-only moments;
- fitted probabilities are blended with 20% transparent softmax probabilities;
- insufficient histories fall back deterministically to the transparent softmax.

This is a **surrogate-label diagnostic**, not an independently discovered regime model and not independent validation of the rule weights. Its probabilities and final coefficients are diagnostic artifacts. Transparent 0–100 scores remain the signal and reporting decision baseline.

## Transition forecasts

The dominant transparent regime supplies the state history. For horizons of 5, 21, and 63 business-day rows (labelled 1 week, 1 month, and 3 months), the system first estimates an add-one/Dirichlet-smoothed empirical matrix:

\[
\widehat P_{ij}^{(h)} =
\frac{N_{ij}^{(h)}+1}{\sum_j(N_{ij}^{(h)}+1)}.
\]

Each forecast-history row uses only transitions whose destination is already known by that row. With sufficient history, a walk-forward multinomial logit conditions on the current state and the ten current factors. Pipeline thresholds are 252 completed origin/destination pairs, 10 destination observations per eligible class, and 63-row refits. Conditional predictions retain a 25% empirical-probability blend; otherwise the empirical posterior is used.

The full-sample transition matrices are descriptive estimates as of the final run date. `transition_forecast_*.csv.gz` is the causal history appropriate for evaluating forecasts. These probabilities inherit the transparent classifier's definition and are not proof of a structural Markov process.

## Research signals

### Absolute signals

The code-defined default universe is `ZW`, `ZC`, `ZS`, `CF`, `NTR`, `MOS`, `DXY`, and `US2Y`. For asset \(i\), available factor contributions and centered regime scores are summed:

\[
r_{it}=\sum_k a_{ik}z_{kt}
+\sum_r c_{ir}(2S_{rt}/100-1),
\qquad
s_{it}=\tanh(r_{it}).
\]

Each \(s_{it}\) lies in `[-1, 1]`; values inside `[-0.05, 0.05]` are labelled flat. Each record carries direction, absolute strength, confidence, expected horizon, current regime, expected next regime, largest measured driver, and an explicit invalidation condition.

The expected-next-regime field is context, not an extra term in the signal equation. Absolute records use the highest 1-month transition probability. Relative-value records currently repeat the current regime because their generator does not consume transition forecasts.

Base confidence rewards configured-factor coverage, row-level breadth, directional agreement, and signal magnitude. It can be reduced by crowding/event-risk inputs. The saved `raw_confidence` preserves that model output. Historical backtest rows use `backtest_confidence`, which multiplies raw confidence by the point-in-time fraction of configured critical series that were both present and within their own staleness limits on that date. The final run-level quality score is applied only to current-date candidates as `current_quality_multiplier`; an endpoint failure cannot retroactively erase or rescale earlier signals. A failed final quality report makes current candidate confidence zero.

The executive dashboard's overall “heuristic conviction” combines current regime activation, median current signal confidence, recent regime stability, and data quality. It is explicitly uncalibrated and is not a success probability; deterministic demo runs cap it at 75%.

### Relative-value signals

The default pairs are `ZW/CF`, `ZC/CF`, `CF/NG`, `CF/NTR`, `NTR/MOS`, `ZW/ZC`, and `ZW/DXY`. For each available pair, the pipeline computes trailing 60-row z-scores for:

- log price ratio;
- log-return spread;
- beta-neutral log-return residual, with a rolling beta clipped to `[-3, 3]`.

The default style is continuation, compressed with `tanh`. A positive result means long the residual (r_L-\beta r_S): the executable weights are positive in the long leg and (-\beta) in the second leg. A negative estimated beta therefore correctly makes both legs point in the same direction. In portfolio construction, the three methods and overlapping absolute signals are combined by confidence-weighted signed evidence at the asset level. Constituent legs then receive asset-level inverse-volatility sizing and constraints. Consequently, the estimated regression hedge ratio describes the research residual but is not guaranteed to survive exactly in the final multi-signal portfolio. These are research spreads, not instructions about contract months, share quantities, or hedge execution.

## Portfolio construction and risk

At each sizing date, the pipeline forms

\[
\text{raw}_i = \frac{s_i\,c_i}{\widehat\sigma_i},
\]

where confidence \(c_i\) is in `[0, 1]` and volatility/covariance use up to 60 trailing return rows. Raw weights are normalized to unit gross, scaled toward the 10% annualized volatility target, then constrained sequentially. Pipeline defaults are:

- maximum position weight: 25%;
- combined grain-and-energy commodity gross: 40%;
- fertilizer-equity gross: 35%;
- default asset-class gross: 60%;
- maximum code-defined factor exposure: 45%;
- maximum gross exposure: 150%;
- maximum absolute net exposure: 35%.

The sequential projection only shrinks existing exposures; it does not introduce a new hedge or reverse a position. Constraints can leave realized expected volatility below target. Factor exposures are simple code-defined research mappings for grain, agricultural inflation, policy, USD, energy, fertilizer, rates, and equity beta—not estimated security betas.

The current risk report includes gross/net/long/short exposure, covariance volatility, historical 95% VaR/CVaR, marginal/component risk, largest risk contributor, factor overlap, the latest **simulated backtest** drawdown, and a separately labelled static replay of today's weights. No realized brokerage/account NAV is ingested, so neither value is labelled a live drawdown.

## Backtest assumptions

The pipeline produces business-daily signal history and the backtester applies the configured rebalance cadence (weekly by default; daily and monthly are also supported) while preserving the actual last decision timestamp in each period. Targets are shifted by one return row before earning P&L. Between target events, holdings drift with realized returns; turnover and costs are charged when the portfolio trades back to a target. Drawdown scaling is based on simulated NAV through the prior row.

The configured evaluation start is preceded by 504 business days of causal warm-up by default. Features, volatility/covariance estimates, and walk-forward models may use that earlier history, while reported P&L and benchmarks are sliced and rebased at the evaluation start.

Default configured costs are applied as return fractions to absolute weight turnover:

- transaction cost: 5 bps;
- slippage: 2 bps;
- commission: 0 bps unless supplied;
- futures roll cost: 3 bps per assumed roll.

Without an explicit roll calendar, roll cost is amortized using four annual rolls for grains and twelve for natural gas. Default drawdown multipliers are 0.80 at 5% loss, 0.50 at 10%, and 0.20 at 15%. The CLI applies signed signal stops, optional configured regime expectations, and the latest simulated backtest drawdown to the published research portfolio; the uncontrolled vector is retained for attribution. This is a research control, not a live-account loss limit. The backtest applies the same drawdown schedule causally using NAV through the prior row.

Market and equity proxies use simple percentage returns. `US2Y` remains a macro/regime and reporting input, but it is deliberately excluded from portfolio P&L because a cash yield change is not a Treasury total return and no duration/DV01 mapping is implemented. Its research signal is therefore non-executable and receives zero tradability confidence. Contract multipliers in `assets.yaml` are not used in P&L. Consequently, the backtest is an exposure-return study, not a futures accounting, financing, margin, or executable-capacity simulation.

Reported metrics include total and annualized return, annualized volatility, Sharpe, Sortino, maximum drawdown, Calmar, active-day win rate, payoff ratio, annualized turnover, average holding period, number of sign-defined trades, tail loss, VaR/CVaR, total cost, and average gross/net exposure. Sharpe and Sortino use annualized arithmetic excess return; CAGR remains the reported annualized return and the numerator of Calmar.

The same net-return history is compared with an equal-weight asset basket, static long of the first return asset (normally `ZW`), and 60-row time-series momentum. Internal benchmarks use the same execution lag, rebalance rule, transaction/slippage/roll assumptions, and drawdown controls, and report their own turnover, trades, and costs. If a simple benchmark has higher Sharpe than the strategy, the report explicitly adds that result to the executive text and strongest evidence against the thesis. This counterevidence is a core output, not a cosmetic appendix.

The statistical regimes and transition models are walk-forward. The fixed transparent rules themselves are not re-estimated in the CLI backtest, so the main simulation is not a nested model-selection study. Reusable `walk_forward_backtest` and split helpers are available for external strategies.

## Lead/lag and event research

Each full pipeline run evaluates the data-supported Q1/Q5/Q6/Q7/Q8 relationships with signed cross-correlations, distributed-lag OLS with Newey–West errors, and nested Granger-style F tests. Positive lag (k) always means (x_{t-k}\rightarrow y_t); causal regression designs reject negative lags. Outputs include sample size, coefficients or correlations, uncertainty statistics, (p)-values, and a ten-question status table that marks missing global-stocks or policy-event tests as blocked rather than silently omitting them.

These diagnostics establish predictive ordering or association only. They do not identify structural causality, and they remain vulnerable to omitted variables, common shocks, data revisions, and multiple testing. The event-study library computes abnormal-return paths and CARs, but the default pipeline does not fabricate the historical export-ban or official-release event history needed to invoke it.

## Scenarios, stresses, monitoring, and reports

Forward scenarios apply first-order decimal asset shocks to today's static weights. Explicit asset shocks take precedence over factor-implied shocks to avoid double counting. Defaults cover base case, bullish grain shock, bearish grain, and stagflation.

Historical stress testing replays today's static weights over configured historical windows only when the return sample overlaps them. It does not recreate the portfolio or information set that would have existed at the historical date.

Model health compares the most recent 63 rows with earlier factor history, warns on a standardized mean shift above 2.5, measures recent dominant-regime stability, tracks turnover, and compares the recent simulated-backtest tail with a disjoint prior sample when enough observations exist. It is not described as live realized-performance monitoring.

The report layer produces:

- an executive dashboard with regime, causal chain, changes, signals, risks, invalidations, catalysts, evidence for/against, and benchmark counterevidence;
- readable and technical Markdown;
- machine-readable JSON and full CSV/JSON diagnostics;
- up to ten source-labelled charts.

Catalyst recurrence dates in the shipped configuration are deterministic research fixtures. Weekend handling is “following business day” based on weekdays only, not a verified holiday/release calendar. They must not be presented as official WASDE, CPI, ENSO, crop-progress, or FOMC dates.

The library event-study module can calculate raw or market-model abnormal returns, CAR paths, terminal event CARs, and grouped mean/median/standard-error/t-statistic/p-value/hit-rate summaries, optionally conditioned on regime. It is not automatically run or written by the CLI.

## Assumptions and known limitations

1. **Demo results have no empirical investment meaning.** Synthetic pulses and correlations are deterministic fixtures designed to exercise the system.
2. **Default live coverage is incomplete.** Yahoo and FRED/ALFRED are wired; required agricultural fundamentals remain demo-only and are omitted—not synthesized—in live mode.
3. **Public market data are proxies.** Yahoo continuous futures are not validated back-adjusted contract series. FRED observations and vendor timestamps remain subject to source availability and methodology.
4. **The transparent weights are economically chosen, not statistically proven.** Independent labels, calibration, sensitivity analysis, and out-of-sample comparison are future work.
5. **The statistical diagnostic learns transparent labels.** It cannot independently confirm the rule-based regime taxonomy.
6. **Scores are activations, not mutually exclusive probabilities.** Only the diagnostic softmax and transition vectors sum to one.
7. **Missing-data rescaling assumes available factors are representative.** Coverage suppression reduces the worst cases, but remaining signals can still be composition-sensitive.
8. **Age is not a regime factor.** Stale values are removed and age is audited, but recency inside the allowed window does not directly reduce the ten-factor score.
9. **Backtest realism is limited.** There is no contract selection, expiry curve, margin, financing, borrow, dividends, taxes, capacity, market impact, or intraday execution model.
10. **Scenario and stress P&L are static and first order.** They omit nonlinear payoffs, changing correlations, liquidity, and dynamic rebalancing.
11. **No causal claim follows from association.** Economic labels and causal-chain text are interpretations of measured factors, not identified structural estimates.
12. **No execution or fiduciary workflow exists.** Human review, instrument validation, independent risk approval, and production controls are required before any investment use.
