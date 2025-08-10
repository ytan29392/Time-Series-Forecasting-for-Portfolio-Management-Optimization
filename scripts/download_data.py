import yfinance as yf
import pandas as pd
tickers = ["SPY", "AAPL", "BTC-USD"]  # Change to your asset (AAPL, BTC-USD, etc.)

for ticker in tickers:
  data = yf.download(ticker, start="2015-01-01", end="2024-12-31")
  data.reset_index(inplace=True)
  data.to_csv(f"data/{ticker}.csv", index=False)

  print(f"Data saved to data/{ticker}.csv")
