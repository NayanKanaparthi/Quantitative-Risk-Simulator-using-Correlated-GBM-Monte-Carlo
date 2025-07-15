import numpy as np
import pandas as pd
import yfinance as yf


def fetch_data(tickers, start="2022-01-01", end="2024-01-01"):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    data = pd.concat([raw[ticker]['Close'] for ticker in tickers], axis=1)
    data.columns = tickers
    return data.dropna()


def calculate_statistics(log_returns):
    mu = log_returns.mean().values * 252
    sigma = log_returns.std().values * np.sqrt(252)
    corr_matrix = log_returns.corr().values
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return mu, sigma, cov_matrix, corr_matrix


def simulate_portfolio(mu, sigma, cov_matrix, weights, T, N, use_corr=True, initial_price=100):
    n_assets = len(weights)
    dt = 1 / 252
    chol = np.linalg.cholesky(cov_matrix) if use_corr else np.diag(sigma)

    sim_paths = np.zeros((T, N))
    for i in range(N):
        prices = np.ones(n_assets) * initial_price
        path = []
        for _ in range(T):
            z = np.random.normal(size=n_assets)
            shocks = chol @ z
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = shocks * np.sqrt(dt)
            prices *= np.exp(drift + diffusion)
            value = np.dot(weights, prices)
            path.append(value)
        sim_paths[:, i] = path

    return sim_paths


def calculate_risk_metrics(sim_paths, initial_price, confidence, investment):
    final_values = sim_paths[-1]
    returns = (final_values - initial_price) / initial_price * 100
    exp_return = np.mean(returns)
    volatility = np.std(returns)
    var = np.percentile(returns, 100 - confidence)
    cvar = returns[returns < var].mean()
    prob_loss = np.mean(returns < 0) * 100
    var_dollar = var / 100 * investment
    cvar_dollar = cvar / 100 * investment
    return returns, exp_return, volatility, var, cvar, prob_loss, var_dollar, cvar_dollar
