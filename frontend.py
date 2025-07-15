import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

from risk_model import (
    fetch_data,
    calculate_statistics,
    simulate_portfolio,
    calculate_risk_metrics,
)

st.set_page_config(page_title="Monte Carlo Portfolio Risk Simulator", layout="wide")

# ---- Sidebar Inputs ----
st.sidebar.title("Portfolio Configuration")

tickers_input = st.sidebar.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
n_assets = len(tickers)

if n_assets == 0:
    st.stop()

default_weights = [round(1 / n_assets, 2)] * n_assets
weights_input = st.sidebar.text_input("Enter portfolio weights (comma separated):", ", ".join(map(str, default_weights)))
weights = np.array([float(w.strip()) for w in weights_input.split(",") if w.strip()])
if len(weights) != n_assets:
    st.sidebar.error(" Number of weights must match number of tickers.")
    st.stop()
weights /= weights.sum()

years = st.sidebar.slider("Investment Horizon (Years)", 1, 10, 1)
T = 252 * years
N = st.sidebar.slider("Number of Simulations", 500, 5000, 1000, 500)
confidence = st.sidebar.slider("Confidence Level for VaR", 90, 99, 95)
use_corr = st.sidebar.checkbox("Use historical correlation", value=True)
investment = st.sidebar.number_input("Total Investment Amount ($)", value=10000, step=1000, min_value=1000)
run = st.sidebar.button("Run Simulation")

# ---- Main App ----
st.title("Quantitative Risk Simulator using Correlated GBM & Monte Carlo")

if run:
    st.markdown("Fetching historical data and running simulations...")

    # Fetch and process
    try:
        data = fetch_data(tickers)
    except:
        st.error("Error fetching data. Check ticker symbols.")
        st.stop()

    log_returns = np.log(data / data.shift(1)).dropna()
    mu, sigma, cov_matrix, corr_matrix = calculate_statistics(log_returns)
    sim_paths = simulate_portfolio(mu, sigma, cov_matrix, weights, T, N, use_corr)

    # Risk metrics
    returns, exp_return, volatility, var, cvar, prob_loss, var_dollar, cvar_dollar = calculate_risk_metrics(
        sim_paths, 100, confidence, investment
    )

    # ---- Metrics Summary ----
    st.subheader("Portfolio Risk Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{exp_return:.2f} %")
    col2.metric("Volatility", f"{volatility:.2f} %")
    col3.metric(f"VaR ({confidence}%)", f"{var:.2f} %")

    col4, col5 = st.columns(2)
    col4.metric("CVaR", f"{cvar:.2f} %")
    col5.metric("Probability of Loss", f"{prob_loss:.2f} %")

    # ---- Interpretation ----
    st.markdown("### Risk Interpretation")
    st.info(
        f"With a {confidence}% confidence level, your portfolio is not expected to lose more than "
        f"{abs(var):.2f}% (i.e., ${abs(var_dollar):,.2f}) on your ${investment:,.0f} investment.\n\n"
        f"If that worst-case 5% happens, the average loss would be {abs(cvar):.2f}% "
        f"(i.e., ${abs(cvar_dollar):,.2f})."
    )

    # ---- Charts ----
    st.subheader("Simulated Portfolio Value Paths")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(sim_paths[:, :50], alpha=0.6)
    ax1.set_title("Simulated Portfolio Paths (First 50)")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig1)

    st.subheader("Distribution of Final Portfolio Returns")
    fig2 = px.histogram(returns, nbins=50, title="Final Portfolio Return Distribution")
    fig2.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"{confidence}% VaR")
    st.plotly_chart(fig2, use_container_width=True)

    if use_corr:
        st.subheader("Historical Correlation Matrix")
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=tickers, yticklabels=tickers, ax=ax3)
        st.pyplot(fig3)

    # ---- CSV Downloads ----
    st.markdown("### Download Results")

    summary_df = pd.DataFrame({
        "Metric": ["Expected Return (%)", "Volatility (%)", f"VaR ({confidence}%)", "CVaR", "Probability of Loss (%)"],
        "Value": [exp_return, volatility, var, cvar, prob_loss]
    })

    returns_df = pd.DataFrame(returns, columns=["Final Portfolio Returns (%)"])
    returns_csv = returns_df.to_csv(index=False).encode("utf-8")
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="portfolio_risk_summary.csv",
            mime="text/csv",
        )

    with col_d2:
        st.download_button(
            label="Download Return Distribution CSV",
            data=returns_csv,
            file_name="portfolio_final_returns.csv",
            mime="text/csv",
        )

    # ---- Hidden Simulated Paths Table ----
    paths_df = pd.DataFrame(sim_paths)
    paths_df.index.name = "Day"
    paths_df.columns = [f"Simulation {i+1}" for i in range(sim_paths.shape[1])]

    show_paths = st.toggle("Show Raw Simulated Paths Table (Advanced)")

    if show_paths:
        st.subheader("Raw Simulated Portfolio Value Paths")
        st.markdown("This table shows the simulated portfolio values for each of the Monte Carlo paths.")
        st.dataframe(paths_df.head(20))  # Only show first 20 rows for performance

        paths_csv = paths_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="Download Full Simulated Paths CSV",
            data=paths_csv,
            file_name="simulated_paths.csv",
            mime="text/csv"
        )
