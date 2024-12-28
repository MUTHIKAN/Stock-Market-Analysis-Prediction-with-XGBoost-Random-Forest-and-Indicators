import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
import matplotlib.pyplot as plt

# Step 1: Fetch historical stock data from Yahoo Finance
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Fetch data for a specific stock (e.g., NIFTY 50)
stock_symbol = '^NSEI'  # NIFTY 50 Index
start_date = '2000-01-01'
end_date = '2024-11-22'
data = fetch_stock_data(stock_symbol, start_date, end_date)

# Display the first and last few rows of the stock data
print(data.head())
print(data.tail())

# Step 2: Generate Example Options Data (Rho, Vega, Theta, Gamma, Delta, IV%) for the same dates
# In a real scenario, this data should be collected from an options pricing API or model.
# Here, we'll generate random values as a placeholder.
np.random.seed(42)
options_data = pd.DataFrame({
    'Rho': np.random.randn(len(data)),
    'Vega': np.random.randn(len(data)),
    'Theta': np.random.randn(len(data)),
    'Gamma': np.random.randn(len(data)),
    'Delta': np.random.randn(len(data)),
    'IV%': np.random.rand(len(data)) * 100  # IV% between 0 and 100
}, index=data.index)

# Step 3: Preprocess data for LSTM
def preprocess_data(data, options_data, window_size):
    # Focus on 'Close' price for prediction and include options data
    close_data = data[['Close']]
    
    # Combine close data and options data
    combined_data = pd.concat([close_data, options_data], axis=1)
    
    # Ensure all column names are strings to avoid the error
    combined_data.columns = combined_data.columns.astype(str)

    # Normalize the combined data (both Close and options data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_scaled = scaler.fit_transform(combined_data)

    # Generate X and y for LSTM input
    X, y = [], []
    for i in range(len(data) - window_size):
        combined_features = combined_scaled[i:i + window_size].flatten()  # Flatten the window data
        target_feature = combined_scaled[i + window_size][0]  # Close price as the target
        
        X.append(combined_features)  # Add to X (input)
        y.append(target_feature)  # Add corresponding target (next day's close price)

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    return X, y, scaler

# Prepare data for LSTM (using 60 days of past data to predict the next day's price)
window_size = 60

# Preprocess data for LSTM model input
X, y, scaler = preprocess_data(data, options_data, window_size)

# Step 4: Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output one value (next day's stock price)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the model
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Step 6: Define the prediction function using user input
def predict_option_achievement(input_data, strike_price):
    # Create a DataFrame from user input with column names matching the training data
    column_names = ['Rho', 'Vega', 'Theta', 'Gamma', 'Delta', 'IV%']
    user_features = pd.DataFrame([input_data], columns=column_names)
    
    # Normalize the user input data by using the same scaler as the training data
    user_features_scaled = scaler.transform(user_features)  # Scale the user input
    
    # Reshape for LSTM input (this is required by the model)
    user_features_scaled = user_features_scaled.reshape(1, user_features_scaled.shape[1], 1)
    
    # Make a prediction using the trained model
    predicted_price = model.predict(user_features_scaled)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    # Check if the predicted price exceeds the strike price
    if predicted_price[0][0] > strike_price:
        return f"The option strike price of {strike_price} will likely be achieved. Predicted price: {predicted_price[0][0]:.2f}"
    else:
        return f"The option strike price of {strike_price} will likely not be achieved. Predicted price: {predicted_price[0][0]:.2f}"

# Step 7: Get input from the user
Rho = 2.08
Vega = 12.56
Theta = 15.02
Gamma = 0.00084    
Delta = 0.51
IV_percent = 15.16
strike_price = 23900

# Combine the user input into a feature vector
user_input_data = [Rho, Vega, Theta, Gamma, Delta, IV_percent]

# Call the prediction function with user inputs
prediction = predict_option_achievement(user_input_data, strike_price)
print(prediction)
