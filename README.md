1. **AlphaEngine: 3-Year Forward Equity Signal Regressor**

AlphaEngine is a machine learning pipeline designed to predict long-horizon, 3-year cross-sectional equity returns.

Rather than forecasting absolute price levels or macroeconomic cycles, the model focuses on relative performance by estimating each asset’s expected excess return versus a benchmark, the S&P 500. This enables direct ranking of investment opportunities and systematic portfolio construction based on conviction.

This repository represents a Minimum Viable Product for a signal-driven portfolio allocation system, combining momentum, risk, and fundamental features into a unified framework.

2. **Objective**

The model answers the following question:

Given information available today, which stocks are most likely to outperform the market over the next roughly 3 years, and by how much?

This reframes stock selection as a continuous regression problem rather than a binary classification task, aligning more closely with real-world portfolio construction.

3. **Data Pipeline**

Inputs:

Historical equity price data via `yfinance`

Fundamental metrics: P/E ratio, earnings yield

Benchmark: SPY / S&P 500 ETF

Target construction:

Forward returns are computed over approximately 3 trading years, or about 756 trading days.

The core target is:

`excess_fwd_3y_return = stock_return - benchmark_return`

This means the model is not simply predicting whether a stock goes up. It is predicting whether a stock outperforms the market over the same forward window.

4. **Methodology: Target Transformation**

Equity returns exhibit heavy tails, particularly on the upside. To stabilize training and reduce the influence of extreme outliers, the target is transformed using a symmetric log function:

`y = sign(x) * log(1 + abs(x))`

This compresses extreme values while preserving directional information.

5. **Methodology: Time-Aware Preprocessing**

To mitigate leakage from overlapping forward returns, the dataset is downsampled to monthly observations using the last trading day of each month.

This reduces autocorrelation and forces the model to learn generalized cross-sectional relationships rather than memorizing overlapping daily noise.

6. **Methodology: Feature Engineering**

Features are constructed across three layers.

Raw features:

Momentum: 6-month and 12-month returns

Risk: 12-month volatility, drawdown

Valuation: P/E ratio, earnings yield

Cross-sectional features:

Within-date percentile ranks for major features

Comparable rankings across time and market regimes

Composite factor:

A key engineered feature combines value and quality:

`quality_value_combo = earnings_yield_rank + low_vol_rank`

Where:

`earnings_yield_rank` captures relative cheapness

`low_vol_rank` captures relative stability

This avoids scale distortions and eliminates instability from negative earnings.

7. **Methodology: Model**

Algorithm: LightGBM Regressor

Objective: Regression on transformed excess returns

Time-based split:

Train: pre-2018

Validation: 2018–2020

Test: 2021 onward

8. **Methodology: Calibration**

Predictions are calibrated using Isotonic Regression.

The calibrator is fitted on validation data only to prevent leakage, then applied to test predictions.

This enforces a monotonic relationship between predicted and realized returns, improving interpretability and aligning predicted magnitudes with observed outcomes.

9. **Notebook Structure**

The project is split into three core stages.

`02_feature_engineering`

Builds raw, cross-sectional, and composite features.

`03_model_training_and_signal_validation`

Trains the LightGBM model.

Applies validation-only isotonic calibration.

Validates ranking quality through prediction buckets.

Exports scored test predictions.

`04_strategy_backtest_and_cohort_analysis`

Translates model scores into portfolio strategies.

Evaluates signal maturation across 1, 3, 6, 12, 24, and 36-month horizons.

Stress-tests contributor concentration and robustness.

10. **Portfolio Construction**

The model signal is deployed through two portfolio construction modes rather than a simple low, medium, and high risk ladder.

Core Alpha:

A diversified rank-weighted basket designed to capture broader, more robust market-beating signal.

Current configuration:

Top 20 stocks by calibrated model score

Rank-based weighting

Intended use: dispersed long-horizon alpha capture

Convex Alpha:

A concentrated high-conviction basket designed to capture public-market power-law winners.

Current configuration:

Top 3 stocks by calibrated model score

Rank-based weighting

Intended use: right-tail / asymmetric upside capture

Weighting scheme:

Rank-based exponential weighting is used:

`weight ∝ rank^power`

This avoids excessive reliance on raw prediction magnitudes while still emphasizing higher-conviction signals.

11. **Results: Out-of-Sample 2021–2022**

The model demonstrates strong cross-sectional ranking performance.

Key signal validation results:

Monotonic realized performance across prediction buckets

Approximately 80% spread between top and bottom prediction quintiles

Positive long-horizon excess returns across both strategy modes

12. **Strategy Backtest Summary**

The backtest evaluates monthly selection cohorts across multiple forward horizons.

Core Alpha behaves as the more robust and dispersed strategy:

Positive excess returns across all tested horizons

Strongest performance over 12–36 month horizons

36-month hit rate near 96% versus SPY

Remains positive even after removing top contributors

Convex Alpha behaves as the concentrated right-tail strategy:

Stronger early separation beginning around the 6-month horizon

Higher 36-month median excess return than Core Alpha

More sensitive to a small number of large winners

Performance degrades sharply when top contributors are removed

These results suggest the model supports two distinct deployment modes:

A diversified Core Alpha basket for more stable long-horizon outperformance

A concentrated Convex Alpha basket for public-market power-law capture

13. **Limitations**

The out-of-sample period is relatively short.

The dataset may contain survivorship bias depending on universe construction.

Transaction costs, slippage, taxes, and turnover are not yet modeled.

The current backtest evaluates monthly selection cohorts, not a full live NAV simulation.

The feature set is intentionally limited for MVP scope.

Concentrated strategies are highly contributor-sensitive.

14. **Future Work**

Full rolling portfolio simulation with overlapping 3-year cohorts

Turnover, transaction cost, and slippage modeling

Sector-neutral and sector-capped portfolio construction

Expanded fundamentals: profitability, growth, balance sheet metrics

Macro regime features: interest rates, VIX, inflation, liquidity conditions

Permutation importance and SHAP analysis

Separate optimization paths for Core Alpha robustness and Convex Alpha right-tail capture

Monthly or quarterly shorter-horizon model variant

15. **Disclaimer**

This project is a research prototype and not an investment strategy. Results are for exploratory and educational purposes only and should not be interpreted as financial advice.

This project is personal work and not related to my employer.

16. **Summary**

This project demonstrates:

End-to-end machine learning pipeline design

Time-aware modeling on financial data

Cross-sectional feature engineering

Validation-only model calibration

Translation of predictive signals into distinct portfolio strategies

Strategy-level robustness testing through cohort backtesting
