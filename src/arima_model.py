from pmdarima import auto_arima
import pandas as pd

def train_arima_model(train_series):
    model = auto_arima(
        train_series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    return model

def forecast_arima(model, test_series):
    n_periods = len(test_series)
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_index = test_series.index
    forecast_series = pd.Series(forecast, index=forecast_index)
    conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=["Lower", "Upper"])
    return forecast_series, conf_int_df
