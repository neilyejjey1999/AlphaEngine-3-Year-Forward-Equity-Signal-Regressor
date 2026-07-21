# AlphaEngine Open: Long-Horizon Equity Signal Research Framework

> **Project status: retired research prototype**
>
> This repository contains the completed first iteration of AlphaEngine: a fully open, reproducible proof of concept built primarily with free market data from `yfinance` and current S&P 500 constituent metadata.
>
> Development of this version has concluded. The next generation of AlphaEngine will be rebuilt in a separate repository around point-in-time market data, historical index membership, delisting and corporate-action outcomes, and expanded fundamentals from the Financial Modeling Prep (FMP) API.
>
> This repository will remain available as the free version of the project. The future FMP-backed version will not distribute licensed market data or API credentials; users who wish to reproduce it will need to provide their own authorized FMP API key.

---

## Overview

AlphaEngine is a machine learning research pipeline designed to predict long-horizon, cross-sectional equity outperformance.

Rather than forecasting absolute stock prices or macroeconomic cycles, the system estimates each stock's expected excess return versus a market benchmark. The resulting signal is used to rank investment opportunities and translate those rankings into systematic portfolio construction rules.

The project began as a minimum viable product for a 3-year forward equity signal. It was later refactored into a bifurcated research framework with two distinct deployment objectives:

1. **Core Alpha** — a diversified basket designed to capture broad, durable market outperformance.
2. **Convex Alpha** — a concentrated basket designed to capture public-market power-law winners.

The central research finding from this version is that model quality cannot be judged only by global signal metrics. A model that is not the strongest standalone ranking signal may still perform best when paired with the portfolio construction policy it was designed to support.

---

## Project Lineage

### AlphaEngine Open — this repository

This version was built to determine how far a long-horizon equity research pipeline could be pushed using free and broadly accessible data.

It demonstrates:

```text
open-data market ingestion
cross-sectional feature engineering
time-aware model training
signal calibration
portfolio construction
cohort backtesting
contributor stress testing
SHAP-based interpretability
current portfolio generation
```

It remains useful as:

```text
a reproducible proof of concept
a research and education project
a baseline for future point-in-time experiments
a demonstration of end-to-end ML system design
```

### Next-generation AlphaEngine — separate repository

The next iteration will be rebuilt as a separate project rather than layered onto this repository.

The planned architecture will use Financial Modeling Prep data to support:

```text
historical S&P 500 membership
former and delisted constituents
acquisitions, bankruptcies, ticker changes, and corporate actions
historical market capitalization
expanded financial statements and ratios
point-in-time feature availability
survivorship-aware target construction
broader model and portfolio experimentation
```

The new repository will contain the full ingestion, modeling, testing, and research code, but it will not redistribute FMP data or credentials. Users will be required to supply their own API key and generate the licensed datasets locally.

---

## Why This Version Is Retired

AlphaEngine Open successfully established that the core research concept is viable, but it also revealed the limits of an entirely free-data implementation.

The current universe is constructed from present-day S&P 500 membership. As a result, historical training rows may exclude companies that were formerly in the index but were later removed, acquired, delisted, or otherwise ceased to exist in their original form.

That creates an important survivorship-bias limitation:

```text
today's surviving constituents
        are not the same as
the investable universe that existed historically
```

The project also lacks a fully authoritative point-in-time source for:

```text
historical market capitalization
historical constituent membership
filing-aware fundamentals
corporate-action outcomes
terminal returns for disappearing securities
```

Further algorithmic complexity would not fully solve those data limitations. The most useful next step is therefore a ground-up rebuild around a stronger point-in-time data foundation.

This repository is being retired because it completed its purpose: it proved the concept, exposed the most important research constraints, and established the architecture that the next version can improve.

---

## Objective

AlphaEngine answers the question:

> Given information available today, which stocks are most likely to outperform the market over the next roughly 3 years, and by how much?

This reframes stock selection as a continuous regression problem rather than a binary classification task.

The model is trained to predict forward excess return:

```text
excess_fwd_3y_return = stock_forward_return - benchmark_forward_return
```

The system is therefore not merely predicting whether a stock rises. It is predicting whether the stock outperforms the market over the same forward window.

---

## Current Research Architecture

The current system separates three layers:

```text
Feature Engineering
        ↓
Signal Generation
        ↓
Portfolio Construction
```

Earlier versions used one feature set and one model. The final open-data version tests multiple signal variants and evaluates them under separate portfolio construction policies.

Current V4 signal variants:

```text
shared_baseline_v4
core_alpha_v4
convex_alpha_v4
```

Current portfolio policies:

```text
core_alpha:
    top 20 stocks
    diversified rank-weighted basket

convex_alpha:
    top 3 stocks
    concentrated rank-weighted basket
```

Notebook 04 evaluates every signal under every portfolio policy. This avoids assuming that a Core-trained model must automatically be deployed through the Core portfolio or that a Convex-trained model must automatically be deployed through the Convex portfolio.

---

## Open-Data Pipeline

Primary inputs:

- Historical equity price data from `yfinance`
- Current S&P 500 universe metadata
- Benchmark ETF data, primarily SPY
- Basic valuation metrics such as P/E ratio and earnings yield
- Current snapshot metadata for live portfolio output and explanation

The pipeline intentionally avoids using current market capitalization as a historical feature.

Current market capitalization is available through snapshot metadata, but a valid historical market-cap feature requires point-in-time shares outstanding or a direct historical market-cap source. Applying today's market capitalization or today's share count across historical rows would introduce leakage.

A separate experimental backfill was later developed using historical shares outstanding, but the next-generation project will use a more complete point-in-time source rather than making that reconstruction the foundation of this repository.

---

## Target Construction

Forward returns are computed over approximately 3 trading years, or about 756 trading days.

The core modeling target is:

```text
excess_fwd_3y_return = stock_return - benchmark_return
```

The transformed training target is:

```text
target = sign(excess_fwd_3y_return) * log(1 + abs(excess_fwd_3y_return))
```

This symmetric log transform compresses extreme winners and losers while preserving direction. It helps stabilize training in a return distribution with heavy tails.

---

## Time-Aware Preprocessing

To reduce leakage and autocorrelation from overlapping forward returns, the modeling dataset is downsampled to monthly observations using the last available trading day of each month.

The model uses a time-based split:

```text
Train:      pre-2018
Validation: 2018–2020
Test:       2021 onward
```

The validation set is used for model calibration. The test set is held out for out-of-sample signal and strategy evaluation.

---

## Feature Engineering

Features are created in `02_feature_engineering`.

The V4 feature set includes the following families.

### Raw Features

```text
6-month return
12-month return
12-month volatility
drawdown
P/E ratio
earnings yield
sector metadata
```

### Cross-Sectional Rank Features

Most raw features are converted into within-date percentile ranks. This makes features more comparable across time and market regimes.

Examples:

```text
ret_6m_rank
ret_12m_rank
vol_12m_rank
drawdown_rank
low_vol_rank
earnings_yield_rank
pe_cheap_rank
```

### Sector-Relative Valuation Features

The V4 pipeline adds sector-relative valuation features:

```text
sector_earnings_yield_rank
sector_pe_cheap_rank
sector_valuation_rank_combo
```

These features compare companies against economically similar peers rather than only against the full market universe.

### Core Alpha Features

Core Alpha features are designed to support diversified, durable, long-horizon outperformance.

Examples:

```text
core_quality_value_combo
core_stability_combo
core_momentum_quality_combo
core_defensive_value_combo
```

Core Alpha emphasizes:

```text
quality
valuation
stability
lower volatility
sector-relative cheapness
```

### Convex Alpha Features

Convex Alpha features are designed to support concentrated right-tail selection.

Examples:

```text
convex_momentum_value_combo
convex_momentum_vol_combo
convex_rebound_combo
convex_value_rebound_combo
```

Convex Alpha emphasizes:

```text
momentum
volatility / optionality
rebound potential
valuation support
right-tail capture
```

---

## Model

The final open-data version uses:

```text
Algorithm: LightGBM Regressor
Objective: Regression on transformed 3-year excess returns
Calibration: Validation-only isotonic regression
```

The model predicts transformed excess returns, which are inverted back into return space and then calibrated.

Calibration uses isotonic regression fitted only on the validation set. This avoids test leakage while improving the interpretability of predicted return magnitudes.

---

## Signal Variants

Notebook 03 trains and evaluates three V4 signal variants.

### `shared_baseline_v4`

Uses the shared V4 feature set. This model performed best as a pure global ranking signal in Notebook 03.

### `core_alpha_v4`

Uses shared features plus Core-specific features. This model is designed for diversified top-20 portfolio construction.

### `convex_alpha_v4`

Uses shared features plus Convex-specific features.

It removes raw P/E from the shared feature set while retaining ranked and composite valuation features, based on earlier ablation findings that raw P/E introduced brittle behavior.

---

## Portfolio Construction

Notebook 04 translates model scores into portfolio strategies.

### Core Alpha

Core Alpha is a diversified rank-weighted portfolio.

Current configuration:

```text
Top K: 20 stocks
Weighting: rank-weighted
Power: 0.6
Objective: diversified long-horizon outperformance
```

Core Alpha is intended to behave more like an actively selected ETF-style basket.

### Convex Alpha

Convex Alpha is a concentrated rank-weighted portfolio.

Current configuration:

```text
Top K: 3 stocks
Weighting: rank-weighted
Power: 0.8
Objective: right-tail / power-law winner capture
```

Convex Alpha is intentionally more concentrated and more sensitive to individual winners.

### Weighting Scheme

Both strategies use rank-based weighting:

```text
weight ∝ rank^power
```

This avoids relying too heavily on raw predicted return magnitudes while still assigning larger weights to higher-conviction names.

---

## Out-of-Sample Results

The current out-of-sample test period evaluates monthly selection cohorts beginning in 2021.

Notebook 03 showed that the shared baseline was the strongest pure signal model based on global ranking metrics such as bucket monotonicity and top-bottom spread.

Notebook 04, however, showed that the specialized models performed best when paired with their intended portfolio construction policies.

### 36-Month Portfolio Results

| Portfolio Policy | Winning Signal | Mean Excess Return | Median Excess Return | Hit Rate |
|---|---:|---:|---:|---:|
| Core Alpha | `core_alpha_v4` | 1.6591 | 0.6791 | 100.0% |
| Convex Alpha | `convex_alpha_v4` | 1.1267 | 0.9613 | 91.7% |

The central result of the V4 research pass is:

> The specialized Core and Convex models did not dominate as global ranking signals, but they performed best when deployed through the portfolio construction policies they were designed to support.

This supports the project's separation between signal validation and portfolio-level evaluation.

These results should be interpreted in the context of the limitations documented below, especially the short test period, overlapping long-horizon cohorts, current-constituent universe construction, and absence of a full live NAV simulation.

---

## Robustness and Stress Testing

Notebook 04 evaluates:

```text
1-month, 3-month, 6-month, 12-month, 24-month, and 36-month forward horizons
top-k sensitivity
contributor concentration
dynamic ex-contributor stress tests
sector exposure
selection frequency
```

Dynamic ex-contributor tests remove the top 1, top 3, and top 5 contributors for each signal and strategy pairing.

This avoids hardcoding prior winners and makes the stress tests strategy-specific.

### Core Alpha Robustness

At the 36-month horizon, `core_alpha_v4` under Core construction produced:

```text
Mean excess return:   1.6591
Median excess return: 0.6791
Hit rate:             100.0%
```

After removing its top contributor:

```text
Mean excess return:   1.2418
Median excess return: 0.6631
Hit rate:             100.0%
```

This suggests Core Alpha is not simply one large winner disguised as a diversified portfolio.

### Convex Alpha Robustness

At the 36-month horizon, `convex_alpha_v4` under Convex construction produced:

```text
Mean excess return:   1.1267
Median excess return: 0.9613
Hit rate:             91.7%
```

After removing its top contributor:

```text
Mean excess return:   0.4870
Median excess return: 0.4863
Hit rate:             75.0%
```

After removing its top 3 contributors:

```text
Mean excess return:   0.0635
Median excess return: -0.2234
Hit rate:             33.3%
```

This confirms that Convex Alpha is highly contributor-sensitive, which is consistent with its concentrated right-tail mandate.

---

## Current Portfolio Generation

Notebook 05 generates current Core Alpha and Convex Alpha portfolio CSVs using the trained V4 models.

Outputs include:

```text
current_core_alpha_portfolio_YYYYMMDD.csv
current_convex_alpha_portfolio_YYYYMMDD.csv
current_core_alpha_shap_details_YYYYMMDD.csv
current_convex_alpha_shap_details_YYYYMMDD.csv
current_portfolio_methodology_YYYYMMDD.md
```

Portfolio outputs include:

```text
ticker
company
sector
sub-industry
portfolio weight
model score
current metadata
SHAP-based reason column
not-financial-advice disclaimer
```

The SHAP reason column explains which features contributed most positively to each holding's score, along with any major negative contributors.

Example structure:

```text
Positive drivers: momentum composite; sector-relative valuation; low-volatility rank
Watchouts: drawdown severity; raw volatility
```

This makes the output more interpretable and closer to an ETF-style holdings report.

---

## Notebook Structure

### `01_data_ingestion`

Downloads and saves:

```text
equity price data
benchmark price data
S&P 500 metadata
current ticker snapshot metadata
```

### `02_feature_engineering`

Builds:

```text
raw return, risk, and valuation features
cross-sectional rank features
sector-relative valuation features
Core Alpha composite features
Convex Alpha composite features
feature family definitions
```

Outputs:

```text
features_v4.parquet
feature_families_v4.json
```

### `03_model_training_and_signal_validation`

Trains:

```text
shared_baseline_v4
core_alpha_v4
convex_alpha_v4
```

Performs:

```text
time-based train, validation, and test split
LightGBM training
validation-only isotonic calibration
prediction bucket validation
signal comparison
model artifact export
```

Outputs scored test predictions for Notebook 04.

### `04_strategy_backtest_and_cohort_analysis`

Evaluates all signal variants under both portfolio policies.

Performs:

```text
multi-horizon cohort backtesting
top-k sensitivity
contributor concentration analysis
dynamic ex-contributor stress tests
sector exposure analysis
promotion decision summary
```

### `05_portfolio_generation_and_interpretability`

Generates current portfolio candidates and explanations.

Outputs:

```text
Core Alpha portfolio CSV
Convex Alpha portfolio CSV
SHAP detail CSVs
methodology note
feature importance summaries
```

---

## Key Findings

This research pass produced several important findings:

1. **The V4 shared feature set improved the base signal.**

   The shared baseline was the strongest pure global ranking model in Notebook 03.

2. **Specialized models were not globally superior.**

   The Core and Convex models did not dominate the shared baseline across all signal-validation metrics.

3. **Specialized models worked best at the portfolio level.**

   When tested under their intended portfolio construction policies, `core_alpha_v4` and `convex_alpha_v4` produced the best 36-month results.

4. **Core Alpha was more robust and diversified.**

   Core Alpha retained strong performance after removing its top contributor.

5. **Convex Alpha was more contributor-sensitive.**

   Convex Alpha depended heavily on a small number of winners, consistent with its right-tail mandate.

6. **Signal validation and portfolio validation should be separated.**

   A globally cleaner ranking model is not always the best model for a specific deployment objective.

7. **Data quality became the binding constraint.**

   The experiment showed that meaningful future improvements require better point-in-time universe, market-cap, fundamental, and corporate-action data rather than model complexity alone.

---

## Limitations

This project is a research prototype and has several important limitations:

```text
The out-of-sample period is relatively short.
The 36-month forward return cohorts overlap.
The universe is based primarily on current S&P 500 constituents and is therefore exposed to survivorship bias.
Former constituents, acquired companies, bankruptcies, and historical removals are not comprehensively represented.
Transaction costs, slippage, taxes, and turnover are not modeled.
The backtest evaluates monthly selection cohorts rather than a full live NAV simulation.
The feature set is limited relative to institutional equity models.
Historical market capitalization is not a fully integrated authoritative feature.
Point-in-time financial statement availability is not comprehensively modeled.
Concentrated strategies are highly contributor-sensitive.
yfinance data may contain missing values, revisions, or vendor inconsistencies.
```

The results should therefore be interpreted as promising research evidence, not as proof of a deployable investment strategy.

---

## Next-Generation Research Direction

Development will continue in a separate repository using a stronger point-in-time data foundation.

Planned research areas include:

```text
historical constituent membership
survivorship-aware universe construction
delisting and acquisition outcome modeling
bankruptcy and terminal-return handling
historical market capitalization
expanded profitability, growth, leverage, margin, and cash-flow features
filing-aware point-in-time fundamentals
ticker and security identity mapping
corporate-action normalization
walk-forward retraining
transaction-cost and turnover modeling
full rolling NAV simulation
sector-neutral and sector-capped construction
liquidity and capacity constraints
benchmark factor models
ensemble and alternative model experiments
portfolio optimization beyond rank weighting
```

The current repository will remain frozen as the open-data baseline against which the next-generation system can be compared.

---

## Reproducibility and Data Access

This repository is designed to remain fully reproducible with free data sources.

The future FMP-backed repository will follow a bring-your-own-key model:

```python
import os
import getpass

FMP_API_KEY = os.getenv("FMP_API_KEY")

if not FMP_API_KEY:
    FMP_API_KEY = getpass.getpass("Enter your FMP API key: ")
```

The public repository will include:

```text
ingestion code
schemas
feature engineering
model training
backtesting
tests
documentation
synthetic example outputs
```

It will not include:

```text
API credentials
raw licensed FMP responses
complete vendor-derived datasets
redistributed historical data panels
```

Users who wish to reproduce the next-generation project will need their own authorized access to the required data.

---

## Disclaimer

This project is a personal research prototype.

It is not financial advice, investment advice, or a recommendation to buy or sell any security. Results are for exploratory and educational purposes only.

Backtested performance is hypothetical and does not account for all real-world trading frictions, including transaction costs, slippage, taxes, liquidity constraints, borrow constraints, or behavioral implementation risk.

This project is personal work and is not related to my employer.

---

## Summary

AlphaEngine Open demonstrates:

```text
end-to-end machine learning pipeline design
time-aware modeling on financial data
cross-sectional feature engineering
target transformation for heavy-tailed returns
validation-only model calibration
signal-family research design
portfolio construction from predictive signals
cohort-based strategy backtesting
dynamic contributor stress testing
SHAP-based portfolio interpretability
current portfolio CSV generation
```

This repository represents the completed first generation of AlphaEngine: a free, open-data proof of concept that pushed an accessible market-data stack close to its practical research limits.

Its most important result is not a single performance statistic. It is the research conclusion that the next meaningful improvement must come from a more complete point-in-time data foundation.

That next stage will continue in a separate repository.
