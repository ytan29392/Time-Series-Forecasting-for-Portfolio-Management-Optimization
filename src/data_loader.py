# import yfinance as yf
# import pandas as pd

# def load_tsla_data(start='2015-01-01', end='2025-08-01'):
#     df = yf.download('TSLA', start=start, end=end)['Adj Close']
#     df = df.asfreq('B').fillna(method='ffill')
#     return df

import yfinance as yf
import pandas as pd

def load_tsla_data(start='2015-01-01', end='2025-08-01'):
    df = yf.download('TSLA', start=start, end=end)
    print(df.head())  # 👈 Add this line
    return df
