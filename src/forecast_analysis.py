import pandas as pd
import matplotlib.pyplot as plt

def forecast_future_arima(model, last_date, steps=252):
    """
    Forecast future values using trained ARIMA model
    """
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
    forecast, confint = model.predict(n_periods=steps, return_conf_int=True)
    
    forecast_series = pd.Series(forecast, index=future_dates)
    confint_df = pd.DataFrame(confint, index=future_dates, columns=["Lower", "Upper"])
    
    return forecast_series, confint_df

def plot_forecast(historical_series, forecast_series, confint_df, title="12-Month Forecast"):
    plt.figure(figsize=(14, 6))
    plt.plot(historical_series[-500:], label='Historical')
    plt.plot(forecast_series, label='Forecast')
    plt.fill_between(confint_df.index, confint_df['Lower'], confint_df['Upper'], color='pink', alpha=0.3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
