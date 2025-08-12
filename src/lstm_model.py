import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- Helper function to create sequences ---
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# --- Prepare Data ---
def prepare_lstm_data(train_series, test_series, seq_length=60):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    
    # Combine train + test for seamless sequence building
    full_data = np.concatenate((train_scaled, scaler.transform(test_series.values.reshape(-1, 1))))
    
    # Only last `len(test)` sequences from combined data
    X_train, y_train = create_sequences(train_scaled, seq_length)
    
    # Prepare input for forecasting
    inputs = full_data[-(len(test_series) + seq_length):]
    X_test = [inputs[i - seq_length:i] for i in range(seq_length, len(inputs))]
    
    return X_train, y_train, np.array(X_test), scaler

# --- Build LSTM Model ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Forecast with LSTM ---
def forecast_lstm(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions).flatten()
