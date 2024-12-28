import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt

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
    rs = gain / loss
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

    # Average True Range (ATR)
    high_low = stock_data['High'] - stock_data['Low']
    high_close = (stock_data['High'] - stock_data['Close'].shift()).abs()
    low_close = (stock_data['Low'] - stock_data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1)
    stock_data['ATR'] = true_range.max(axis=1).rolling(window=14).mean()

    # Stochastic Oscillator (Stoch)
    stock_data['Stoch_K'] = 100 * (stock_data['Close'] - stock_data['Low'].rolling(window=14).min()) / (stock_data['High'].rolling(window=14).max() - stock_data['Low'].rolling(window=14).min())
    stock_data['Stoch_D'] = stock_data['Stoch_K'].rolling(window=3).mean()

    # On-Balance Volume (OBV)
    stock_data['OBV'] = (stock_data['Volume'] * ((stock_data['Close'] > stock_data['Close'].shift()).astype(int) * 2 - 1)).cumsum()

    # Commodity Channel Index (CCI)
    stock_data['Typical_Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    stock_data['CCI'] = (stock_data['Typical_Price'] - stock_data['Typical_Price'].rolling(window=20).mean()) / (0.015 * stock_data['Typical_Price'].rolling(window=20).std())

    # Momentum Indicator (Rate of Change)
    stock_data['Momentum'] = stock_data['Close'].diff(4)  # 4-period momentum

    # Drop rows with missing values after all calculations
    stock_data.dropna(inplace=True)

    return stock_data

# Function to create lag features
def create_lag_features(stock_data, lags=5):
    for lag in range(1, lags+1):
        stock_data[f'Close_Lag_{lag}'] = stock_data['Close'].shift(lag)
    stock_data.dropna(inplace=True)  # Drop rows with missing values
    return stock_data

# Function to prepare the data for prediction
def prepare_data(stock_data):
    # Use technical indicators and lag features as predictors
    feature_columns = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal', 
                       'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'Momentum'] + [f'Close_Lag_{i}' for i in range(1, 6)]
    X = stock_data[feature_columns]
    y_high = stock_data['High']
    y_low = stock_data['Low']
    y_close = stock_data['Close']
    y_open = stock_data['Open']
    return X, y_high, y_low, y_close, y_open

# Function to train XGBoost models for High, Low, and Close, Open prices
def train_xgboost_models(X, y_high, y_low, y_close, y_open):
    # Convert X to float32 for compatibility with XGBoost
    X = X.astype('float32')

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X)

    # Train models for High, Low, and Close predictions
    model_high = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5)
    model_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5)
    model_close = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5)
    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5)
    
    model_high.fit(dmatrix, y_high)
    model_low.fit(dmatrix, y_low)
    model_close.fit(dmatrix, y_close)
    model_open.fit(dmatrix, y_open)
    
    return model_high, model_low, model_close, model_open


# Function to train XGBoost models for High, Low, and Close, Open prices
def train_xgboost_models(X, y_high, y_low, y_close, y_open):
    # Train models for High, Low, and Close predictions
    model_high = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5, eval_metric='rmse')
    model_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5, eval_metric='rmse')
    model_close = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5, eval_metric='rmse')
    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.01, max_depth=5, eval_metric='rmse')
    
    model_high.fit(X, y_high)
    model_low.fit(X, y_low)
    model_close.fit(X, y_close)
    model_open.fit(X, y_open)
    
    return model_high, model_low, model_close, model_open


# Streamlit UI and further code remains unchanged

st.title('Indian Stock Price Prediction with Muthikan')
st.write('Stock market prediction leverages data and analytics to forecast price movements and trends for informed investment decisions.')

# Function to load stock symbols from an Excel file
def load_stock_symbols_from_file(file_path, column_name='B'):
    try:
        # Read the Excel file
        stock_df = pd.read_excel(file_path)
        # Extract symbols from the specified column
        stock_symbols = stock_df[column_name].dropna().unique().tolist()
        return stock_symbols
    except Exception as e:
        st.error(f"Error loading stock symbols: {e}")
        return []

# Specify the path to your Excel file
file_path = "C:\\Users\\Muthikan\\Desktop\\Stock Market Project\\stocklist.xlsx"
 
# Load stock symbols and add them to the list
stock_list = load_stock_symbols_from_file(file_path, column_name='Symble')
stock_list = list(set(stock_list))  # Remove duplicates if any

# Streamlit Dropdown with updated stock list
ticker = st.selectbox('Select Stock Ticker:', stock_list)

# Store data in session_state to prevent reload
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'model_high' not in st.session_state:
    st.session_state.model_high = None
if 'model_low' not in st.session_state:
    st.session_state.model_low = None
if 'model_close' not in st.session_state:
    st.session_state.model_close = None
if 'model_open' not in st.session_state:
    st.session_state.model_open = None

# Fetch stock data and train models only when needed
if ticker != "" and st.button('Get Stock Data'):
    st.write(f"Fetching data for {ticker}...")  
    stock_data = get_stock_data(ticker)
    
    # Display the stock data
    st.write(stock_data.tail())

    # Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)

    # Create lag features
    stock_data = create_lag_features(stock_data, lags=5)

    # Prepare data for training
    X, y_high, y_low, y_close,y_open = prepare_data(stock_data)

    # Train the XGBoost models
    model_high, model_low, model_close,model_open = train_xgboost_models(X, y_high, y_low, y_close,y_open)

    # Store the trained models and data in session_state
    st.session_state.stock_data = stock_data
    st.session_state.model_high = model_high
    st.session_state.model_low = model_low
    st.session_state.model_close = model_close
    st.session_state.model_open = model_open

# Allow the user to input the Open Price
if st.session_state.stock_data is not None:
    # User input for Open price
    open_price_input = st.number_input('Enter Open Price:', min_value=0.0, step=0.1, format="%.2f")

    if st.button('Predict Today'):
        if open_price_input > 0:
            # Prepare the input data for prediction (using the last row as base)
            last_row = st.session_state.stock_data.iloc[-1:].copy()
            last_row['Open'] = open_price_input  # Update Open price

            # Generate the feature set for prediction
            features = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal', 
                        'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'Momentum'] + [f'Close_Lag_{i}' for i in range(1, 6)]

            # Create the feature matrix for prediction (using the last row of data, including lagged features)
            feature_data = last_row[features].astype('float32')

            # Make sure that we pass the feature data as a numpy array or DataFrame, not DMatrix
            feature_array = np.array(feature_data)  # Convert to numpy array for prediction

            # Make predictions using the trained models
            predicted_high = st.session_state.model_high.predict(feature_array)[0]
            predicted_low = st.session_state.model_low.predict(feature_array)[0]
            predicted_close = st.session_state.model_close.predict(feature_array)[0]

            # Prepare data for the line chart
            last_30_days = st.session_state.stock_data[-30:]

            predicted_data = pd.DataFrame({
                'Open': [open_price_input] * 2,
                'High': [predicted_high] * 2,
                'Low': [predicted_low] * 2,
                'Close': [predicted_close] * 2,
                'Date': [last_30_days.index[-1], last_30_days.index[-1] + pd.Timedelta(days=1)]
            })

        
        # Provide a textual explanation of the chart
        st.write("""
        The line chart above shows the **Open**, **High**, **Low**, and **Close** prices for the last 7 days along with the predicted values for the next day.
        The predicted values are based on the model trained using historical stock data and technical indicators.
        Use this information for decision-making in the stock market.
        """)
        
        # Plot the historical and predicted data on the line chart
        st.line_chart(pd.concat([last_30_days[['Open', 'High', 'Low', 'Close']], predicted_data.set_index('Date')[['Open', 'High', 'Low', 'Close']]]))

        # Optionally, display some more insights or predictions for other days if needed
        st.write("Here are the predicted prices for the Today day (following your Open Price input):")
        st.write(f"- Predicted High: ₹{predicted_high:.2f}")
        st.write(f"- Predicted Low: ₹{predicted_low:.2f}")
        st.write(f"- Predicted Close: ₹{predicted_close:.2f}")

        

 # Function to predict the next 7 days
def predict_next_7_days(models, stock_data, lags=5):
    # Unpack the models
    model_high, model_low, model_close, model_open = models

    # Prepare a DataFrame to store future predictions
    future_predictions = []
    last_row = stock_data.iloc[-1:].copy()  # Start with the last available data

    for day in range(2, 9):  # Predict for the next 7 days
        # Update lag features
        for lag in range(1, lags + 1):
            last_row[f'Close_Lag_{lag}'] = stock_data['Close'].iloc[-lag]

        # Generate features for prediction
        features = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal', 
                       'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'Momentum'] + [f'Close_Lag_{i}' for i in range(1, 6)]
        feature_data = last_row[features].astype('float32')

            # Make sure that we pass the feature data as a numpy array or DataFrame, not DMatrix
        feature_array = np.array(feature_data)
        # Predict High, Low, and Close
        predicted_high = model_high.predict(feature_array)[0]
        predicted_low = model_low.predict(feature_array)[0]
        predicted_close = model_close.predict(feature_array)[0]
        predicted_open = model_open.predict(feature_array)[0]

        # Append the predictions
        future_predictions.append({
            'Date': last_row.index[0] + pd.Timedelta(days=day),
            'Open':predicted_open,
            'High': predicted_high,
            'Low': predicted_low,
            'Close': predicted_close
        })

        # Update the last_row for the next prediction
        last_row['Close'] = predicted_close  # Use predicted Close as the new Close
        stock_data = pd.concat([stock_data, pd.DataFrame({'Close': [predicted_close]}, index=[last_row.index[0] + pd.Timedelta(days=day)])])

    return pd.DataFrame(future_predictions).set_index('Date')

# Update Streamlit for the next 7 days prediction
if st.session_state.stock_data is not None:
    if st.button('Predict Next 7 Days'):
        # Predict next 7 days
        future_data = predict_next_7_days(
            models=(st.session_state.model_high, st.session_state.model_low, st.session_state.model_close,st.session_state.model_open),
            stock_data=st.session_state.stock_data
        )

        # Combine historical and future data for visualization
        combined_data = pd.concat([
            st.session_state.stock_data[['Close']].iloc[-365:],  # Last 30 days
            future_data[['Open','High', 'Low', 'Close']]
        ])

        # Display future predictions
        st.write("### Next 7 Days Predictions")
        st.write(future_data)

        # Plot the combined data
        st.line_chart(combined_data)
 