# Quantitative Risk Simulator using Correlated GBM & Monte Carlo

This project is an interactive risk simulation tool designed to model and analyze the behavior of multi-asset portfolios. It uses Geometric Brownian Motion (GBM) and Monte Carlo simulation to estimate financial risk, incorporating historical asset correlations via Cholesky decomposition.

Built with Python and Streamlit, the simulator allows users to explore Value at Risk (VaR), Conditional Value at Risk (CVaR), probability of loss, and final return distributions in a clear and interpretable interface.

---

<details>
<summary><strong>Features</strong></summary>

- Simulates asset price paths using correlated Geometric Brownian Motion
- Supports multi-asset portfolios with user-defined weightings
- Historical correlation handled through Cholesky decomposition
- Monte Carlo simulation over configurable time horizons and number of trials
- Computes key portfolio risk metrics:
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Probability of loss
  - Volatility and expected return
- Visualizations of:
  - Simulated portfolio value paths
  - Return distribution with risk thresholds
  - Historical correlation matrix
- Allows CSV downloads of:
  - Risk metric summary
  - Final return distribution
  - Raw simulation paths (optional)
- Risk interpretation module for non-technical users

</details>

---

<details>
<summary><strong>Installation</strong></summary>

Clone the repository and install dependencies:

```bash
git clone https://github.com/NayanKanaparthi/Quantitative-Risk-Simulator-using-Correlated-GBM-Monte-Carlo.git
cd Quantitative-Risk-Simulator-using-Correlated-GBM-Monte-Carlo
pip install -r requirements.txt


<details>
<summary><strong>Running the Application</strong></summary>

To launch the Streamlit app locally:

```bash
streamlit run app.py
