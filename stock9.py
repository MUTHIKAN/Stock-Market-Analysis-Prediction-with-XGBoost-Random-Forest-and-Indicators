import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker, start_date="2020-01-01", end_date=datetime.today().strftime('%Y-%m-%d')):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate technical indicators
def calculate_technical_indicators(stock_data):
    # Moving Averages
    stock_data['5_day_MA'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['10_day_MA'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['50_day_MA'] = stock_data['Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Safely calculate RSI with handling for zero loss
    rs = gain / loss
    rs[loss.eq(0)] = 0  # If any value of loss is zero, set RSI to 0 to avoid division by zero
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    stock_data['5_day_STD'] = stock_data['Close'].rolling(window=5).std()  # Standard Deviation
    stock_data['Bollinger_High'] = stock_data['5_day_MA'] + (2 * stock_data['5_day_STD'])
    stock_data['Bollinger_Low'] = stock_data['5_day_MA'] - (2 * stock_data['5_day_STD'])

    # Moving Average Convergence Divergence (MACD)
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with missing values after all calculations
    stock_data.dropna(inplace=True)

    return stock_data

# Function to create lag features
def create_lag_features(stock_data, lags=5):
    for lag in range(1, lags + 1):
        stock_data[f'Close_Lag_{lag}'] = stock_data['Close'].shift(lag)
    stock_data.dropna(inplace=True)  # Drop rows with missing values
    return stock_data

# Function to prepare the data for prediction
def prepare_data(stock_data):
    # Use technical indicators and lag features as predictors
    feature_columns = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal'] + [f'Close_Lag_{i}' for i in range(1, 6)]
    X = stock_data[feature_columns]
    y_high = stock_data['High']
    y_low = stock_data['Low']
    y_close = stock_data['Close']
    return X, y_high, y_low, y_close

# Function to train XGBoost models for High, Low, and Close prices
def train_xgboost_models(X, y_high, y_low, y_close):
    # Train models for High, Low, and Close predictions
    model_high = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=5)
    model_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=5)
    model_close = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=5)
    
    model_high.fit(X, y_high)
    model_low.fit(X, y_low)
    model_close.fit(X, y_close)
    
    return model_high, model_low, model_close

# Streamlit App Layout
st.title("Indian Stock Prediction App")
st.sidebar.header("Stock Prediction")

# List of Indian stock tickers (including Nifty50)
stock_list = [
    'TCS.NS', 'RELIANCE.NS', 'HDFC.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
    'HCLTECH.NS', 'LT.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'MARUTI.NS', 'M&M.NS', 
    'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ASIANPAINT.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 
    'HDFCBANK.NS', 'WIPRO.NS', 'POWERGRID.NS', 'SUNPHARMA.NS', 'DIVISLAB.NS', 'RELIANCE.NS', 'INDUSINDBK.NS',
    'TECHM.NS', 'UPL.NS', 'BHEL.NS', 'GAIL.NS', 'IOC.NS', 'ONGC.NS', 'JSWSTEEL.NS', '^NSEI'  # Correct Nifty50 ticker symbol
]

ticker = st.sidebar.selectbox('Select Stock Ticker:', stock_list)

# Radio button for Trading Type (Intra-day or Long-term)
trading_type = st.sidebar.radio("Select Trading Type", ("Intra-day Trading", "Long-term Trading"))

# If a ticker is selected, fetch data and train models
if ticker:
    stock_data = get_stock_data(ticker)
    stock_data = calculate_technical_indicators(stock_data)
    stock_data = create_lag_features(stock_data)
    X, y_high, y_low, y_close = prepare_data(stock_data)
    model_high, model_low, model_close = train_xgboost_models(X, y_high, y_low, y_close)
    
    st.write(f"Latest stock data for {ticker}:")
    st.write(stock_data.tail())

    predicted_closes = []  # Store the predicted closing prices

    if trading_type == "Intra-day Trading":
        # Allow the user to input Open Price
        open_price_input = st.number_input("Enter Open Price for Prediction:", min_value=0.0, step=0.1)
        
        if open_price_input > 0:
            # Prepare the features with the user input
            last_row = stock_data.iloc[-1:].copy()
            last_row['Open'] = open_price_input
            
            # Update lag features based on the latest 'Close' price
            for lag in range(1, 6):
                last_row[f'Close_Lag_{lag}'] = stock_data['Close'].iloc[-lag]

            # Prepare the features for prediction
            features = last_row[['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal'] + [f'Close_Lag_{i}' for i in range(1, 6)]]

            # Make predictions
            predicted_high = model_high.predict(features)[0]
            predicted_low = model_low.predict(features)[0]
            predicted_close = model_close.predict(features)[0]

            st.write(f"Predicted High Price: ₹{predicted_high:.2f}")
            st.write(f"Predicted Low Price: ₹{predicted_low:.2f}")
            st.write(f"Predicted Close Price: ₹{predicted_close:.2f}")

            predicted_closes.append(predicted_close)  # Add the predicted close to the list

    elif trading_type == "Long-term Trading":
        # Long-term Trading: User selects a future date to predict the stock price
        future_date = st.date_input("Select Future Date for Prediction:", min_value=datetime.today())

        if future_date:
            # Convert datetime.today() to date to match the type of future_date
            days_ahead = (future_date - datetime.today().date()).days
            st.write(f"Predictions for {future_date} ({days_ahead} days ahead):")
            
            last_row = stock_data.iloc[-1:].copy()
            
            # Create lag features for future prediction (use latest data)
            for lag in range(1, 6):
                last_row[f'Close_Lag_{lag}'] = stock_data['Close'].iloc[-lag]
            
            # Make future price predictions
            future_features = last_row[['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal'] + [f'Close_Lag_{i}' for i in range(1, 6)]]
            predicted_high = model_high.predict(future_features)[0]
            predicted_low = model_low.predict(future_features)[0]
            predicted_close = model_close.predict(future_features)[0]

            st.write(f"Predicted High Price for {future_date}: ₹{predicted_high:.2f}")
            st.write(f"Predicted Low Price for {future_date}: ₹{predicted_low:.2f}")
            st.write(f"Predicted Close Price for {future_date}: ₹{predicted_close:.2f}")

            # Add the predicted close to the list (even for long-term)
            predicted_closes.append(predicted_close)

    # Line plot of Closing Price over Time
    st.subheader("Stock Closing Price Over Time")

    # Plot the actual closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Close Price', color='blue')

    # If there are predictions, plot them as well
    if predicted_closes:
        # Extend the predictions to the same length as the data
        predicted_dates = pd.date_range(start=stock_data.index[-1], periods=len(predicted_closes) + 1, freq='B')[1:]
        plt.plot(predicted_dates, predicted_closes, label='Predicted Close Price', linestyle='--', color='red')

    plt.title(f'{ticker} Stock Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price (₹)')
    plt.legend()
    st.pyplot(plt)

    # Correlation Heatmap of Technical Indicators
    st.subheader("Correlation Heatmap of Technical Indicators")
    correlation_matrix = stock_data[['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal'] + [f'Close_Lag_{i}' for i in range(1, 6)]].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f"Correlation Heatmap for {ticker}")
    st.pyplot(plt)

# Flowchart for the Prediction Process
st.subheader("Prediction Flowchart")

# Create a flowchart diagram using Graphviz
flowchart = """
digraph G {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor=lightblue];

    Start -> "Get Stock Data" -> "Calculate Technical Indicators" -> "Create Lag Features" -> "Prepare Data" -> "Train XGBoost Models"
    "Train XGBoost Models" -> "User Input (Open Price)"
    "User Input (Open Price)" -> "Make Predictions"
    "Make Predictions" -> End;

    Start [shape=circle, width=0.15, label="Start", fillcolor=lightgreen];
    End [shape=circle, width=0.15, label="End", fillcolor=lightgreen];
}
"""

# Render the flowchart
st.graphviz_chart(flowchart)
