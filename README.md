# AlphaEngine: Long-Horizon Equity Signal Research Framework

AlphaEngine is a machine learning research pipeline designed to predict long-horizon, cross-sectional equity outperformance.

Rather than forecasting absolute stock prices or macroeconomic cycles, the system estimates each stock’s expected excess return versus a market benchmark. This allows the model to rank investment opportunities and translate those rankings into systematic portfolio construction rules.

The project began as a minimum viable product for a 3-year forward equity signal. It has since been refactored into a bifurcated research framework with two distinct deployment objectives:

1. **Core Alpha** — a diversified basket designed to capture broad, durable market outperformance.
2. **Convex Alpha** — a concentrated basket designed to capture public-market power-law winners.

The key research finding from the current version is that model quality cannot be judged only by global signal metrics. A model that is not the strongest pure ranking signal may still perform best when paired with the portfolio construction policy it was designed for.

---

## Objective

AlphaEngine answers the question:

> Given information available today, which stocks are most likely to outperform the market over the next roughly 3 years, and by how much?

This reframes stock selection as a continuous regression problem rather than a binary classification task.

The model is trained to predict forward excess return:

```text
excess_fwd_3y_return = stock_forward_return - benchmark_forward_return
```

This means the model is not simply predicting whether a stock rises. It is predicting whether a stock outperforms the market over the same forward window.

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

Earlier versions used one feature set and one model. The current version tests multiple signal variants and evaluates them under separate portfolio construction policies.

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

Notebook 04 evaluates every signal under every portfolio policy. This prevents the project from assuming that a Core-trained model must be used for Core Alpha or that a Convex-trained model must be used for Convex Alpha.

---

## Data Pipeline

Inputs:

- Historical equity price data from `yfinance`
- S&P 500 universe metadata
- Benchmark ETF data, primarily SPY
- Basic valuation metrics such as P/E ratio and earnings yield
- Current snapshot metadata for live portfolio output and explanation

The pipeline intentionally avoids using current market capitalization as a historical feature. Current market cap is available through snapshot metadata, but historical market cap requires point-in-time share count data. Using today’s market cap or today’s shares outstanding across historical rows would introduce leakage.

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

The V4 feature set includes:

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

The V4 pipeline also adds sector-relative valuation features:

```text
sector_earnings_yield_rank
sector_pe_cheap_rank
sector_valuation_rank_combo
```

These features help compare companies against economically similar peers rather than across the full market universe.

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

The current model uses:

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

Uses shared features plus Convex-specific features. It removes raw P/E from the shared feature set while retaining ranked and composite valuation features, based on earlier ablation findings that raw P/E introduced brittle behavior.

---

## Portfolio Construction

Notebook 04 translates model scores into portfolio strategies.

The project uses two deployment policies.

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

However, Notebook 04 showed that the specialized models performed best when paired with their intended portfolio construction policies.

### 36-Month Portfolio Results

| Portfolio Policy | Winning Signal | Mean Excess Return | Median Excess Return | Hit Rate |
|---|---:|---:|---:|---:|
| Core Alpha | `core_alpha_v4` | 1.6591 | 0.6791 | 100.0% |
| Convex Alpha | `convex_alpha_v4` | 1.1267 | 0.9613 | 91.7% |

This is the central result of the V4 research pass:

> The specialized Core and Convex models did not dominate as global ranking signals, but they did perform best when deployed through the portfolio construction rules they were designed for.

That supports the project’s separation between signal validation and portfolio-level evaluation.

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

Dynamic ex-contributor tests remove the top 1, top 3, and top 5 contributors for each signal/strategy pairing.

This avoids hardcoding prior winners and makes the stress tests strategy-specific.

### Core Alpha Robustness

Core Alpha remains more diversified and resilient.

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

This suggests Core Alpha is not simply one large winner disguised as a portfolio.

### Convex Alpha Robustness

Convex Alpha behaves like a concentrated power-law strategy.

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

This confirms that Convex Alpha is highly contributor-sensitive, which is expected for a concentrated right-tail strategy.

---

## Current Portfolio Generation

Notebook 05 has been reworked into a portfolio generation and interpretability notebook.

It can generate current Core Alpha and Convex Alpha portfolio CSVs using the trained V4 models.

Outputs include:

```text
current_core_alpha_portfolio_YYYYMMDD.csv
current_convex_alpha_portfolio_YYYYMMDD.csv
current_core_alpha_shap_details_YYYYMMDD.csv
current_convex_alpha_shap_details_YYYYMMDD.csv
current_portfolio_methodology_YYYYMMDD.md
```

The portfolio CSVs include:

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

The SHAP reason column explains which features most positively contributed to each holding’s score, along with any major negative contributors.

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
raw return/risk/valuation features
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
time-based train/validation/test split
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

4. **Core Alpha is more robust and diversified.**

   Core Alpha retained strong performance after removing its top contributor.

5. **Convex Alpha is more contributor-sensitive.**

   Convex Alpha depends heavily on a small number of winners, consistent with its right-tail mandate.

6. **Signal validation and portfolio validation should be separated.**

   A globally cleaner ranking model is not always the best model for a specific deployment objective.

---

## Limitations

This project is a research prototype and has several important limitations:

```text
The out-of-sample period is relatively short.
The 36-month forward return cohorts overlap.
The dataset may contain survivorship bias depending on universe construction.
Transaction costs, slippage, taxes, and turnover are not yet modeled.
The current backtest evaluates monthly selection cohorts, not a full live NAV simulation.
The feature set is still limited relative to institutional equity models.
Historical market capitalization is not included because a point-in-time source is not yet available.
Concentrated strategies are highly contributor-sensitive.
yfinance data may contain missing values, revisions, or vendor inconsistencies.
```

The results should therefore be interpreted as promising research evidence, not as proof of a deployable investment strategy.

---

## Future Work

Planned extensions:

```text
Full rolling NAV simulation
Overlapping cohort capital-path modeling
Turnover and transaction cost modeling
Slippage and tax assumptions
Sector-neutral and sector-capped portfolio construction
Historical point-in-time market cap integration
Expanded fundamentals: growth, profitability, margins, leverage, free cash flow
Balance sheet and cash flow statement features
Macro regime features: rates, inflation, VIX, liquidity
Walk-forward retraining
Universe survivorship-bias correction
Separate model objectives for Core and Convex Alpha
Portfolio optimization beyond rank weighting
Monthly or quarterly shorter-horizon model variants
```

---

## Disclaimer

This project is a personal research prototype.

It is not financial advice, investment advice, or a recommendation to buy or sell any security. Results are for exploratory and educational purposes only.

Backtested performance is hypothetical and does not account for all real-world trading frictions, including transaction costs, slippage, taxes, liquidity constraints, borrow constraints, or behavioral implementation risk.

This project is personal work and is not related to my employer.

---

## Summary

AlphaEngine demonstrates:

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

The current version reframes AlphaEngine from a single-model MVP into a broader equity signal research framework that separates feature engineering, signal generation, portfolio construction, and interpretability.
