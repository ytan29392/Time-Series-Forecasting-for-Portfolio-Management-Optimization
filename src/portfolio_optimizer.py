import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Helper: Calculate Portfolio Return, Volatility, Sharpe Ratio ===
def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.015):
    weights = np.array(weights)
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

# === Objective: Negative Sharpe Ratio (for max Sharpe optimization) ===
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.015):
    return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# === Constraint: Sum of weights = 1 ===
def weight_constraint(weights):
    return np.sum(weights) - 1

# === Generate Efficient Frontier ===
def generate_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.015):
    results = []
    weights_list = []

    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results.append([ret, vol, sharpe])
        weights_list.append(weights)

    results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])
    weights_df = pd.DataFrame(weights_list, columns=mean_returns.index)
    return results_df, weights_df

# === Find Optimal Portfolios ===
def optimize_portfolios(mean_returns, cov_matrix, risk_free_rate=0.015):
    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    # Max Sharpe
    opt_sharpe = minimize(negative_sharpe,
                          initial_weights,
                          args=(mean_returns, cov_matrix, risk_free_rate),
                          method='SLSQP',
                          bounds=bounds,
                          constraints={'type': 'eq', 'fun': weight_constraint})

    # Min Volatility
    opt_vol = minimize(lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
                       initial_weights,
                       method='SLSQP',
                       bounds=bounds,
                       constraints={'type': 'eq', 'fun': weight_constraint})

    return opt_sharpe, opt_vol

# === Plot Efficient Frontier ===
def plot_efficient_frontier(results_df, opt_sharpe, opt_vol, mean_returns):
    plt.figure(figsize=(12, 6))
    plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')

    # Plot optimal portfolios
    plt.scatter(opt_sharpe.fun * -1, portfolio_stats(opt_sharpe.x, mean_returns, cov_matrix)[1], c='red', label='Max Sharpe')
    plt.scatter(portfolio_stats(opt_vol.x, mean_returns, cov_matrix)[0], portfolio_stats(opt_vol.x, mean_returns, cov_matrix)[1], c='blue', label='Min Volatility')

    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
