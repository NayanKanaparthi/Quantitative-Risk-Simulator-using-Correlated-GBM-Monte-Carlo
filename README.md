# Quantitative Risk Simulator using Correlated GBM & Monte Carlo

This project is a web-based portfolio risk analysis tool built using Python and Streamlit. It simulates the future performance of a multi-asset portfolio using Geometric Brownian Motion (GBM) and Monte Carlo simulation. The tool supports correlation modeling between assets using Cholesky decomposition and provides key risk metrics to assess downside exposure.

## Overview

The simulator allows users to:

- Configure a portfolio of stocks with custom weights
- Simulate thousands of possible future price paths using historical volatility and drift
- Incorporate historical correlation between assets
- Analyze the distribution of returns over a chosen investment horizon

## Key Features

- Multi-asset Monte Carlo simulation using Geometric Brownian Motion
- Correlated asset price modeling using Cholesky decomposition
- Risk metrics calculated from simulated outcomes:
  - Expected return
  - Volatility
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Probability of loss
- Visualizations:
  - Simulated price paths
  - Histogram of final portfolio returns with VaR cutoff
  - Correlation matrix heatmap
- Export functionality:
  - Risk summary CSV
  - Raw simulation paths CSV
  - Final return distribution CSV

## Methodology

1. Daily log returns are computed from historical data using Yahoo Finance.
2. Annualized drift and volatility are derived for each asset.
3. Correlation matrix is calculated and Cholesky-decomposed to generate correlated random shocks.
4. GBM is simulated for each asset over the selected time horizon and number of simulations.
5. Portfolio value is computed at each time step using user-defined weights.
6. Risk metrics are extracted from the distribution of final simulated returns.

## Use Cases

- Visualizing the effect of asset correlation on portfolio risk
- Comparing portfolios across different investment horizons
- Demonstrating downside risk using probabilistic models
- Educational tool for understanding GBM, Monte Carlo simulation, and VaR

## License

This project is licensed under the MIT License.

## Author

Nayan Kanaparthi  
GitHub: [NayanKanaparthi](https://github.com/NayanKanaparthi)
