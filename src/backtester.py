# src/backtester.py

import pandas as pd
import numpy as np

def compute_portfolio_returns(price_df, weights):
    """
    Compute daily portfolio returns from asset prices and weights
    """
    returns = price_df.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def backtest_strategy(price_df, strategy_weights, benchmark_weights, start='2024-08-01', end='2025-07-31'):
    """
    Run a simple backtest comparing strategy to benchmark over given period.
    """
    price_df = price_df.loc[start:end]
    strategy_returns = compute_portfolio_returns(price_df, strategy_weights)
    benchmark_returns = compute_portfolio_returns(price_df, benchmark_weights)
    
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    return strategy_returns, benchmark_returns, strategy_cum, benchmark_cum
