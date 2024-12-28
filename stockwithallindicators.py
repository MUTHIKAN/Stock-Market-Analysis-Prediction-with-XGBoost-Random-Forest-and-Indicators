import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
    
    # Money Flow Index (MFI)
    typical_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    money_flow = typical_price * stock_data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=14).sum()
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    stock_data['MFI'] = mfi

    # Chaikin Money Flow (CMF)
    money_flow_volume = (stock_data['Close'] - stock_data['Low'] - (stock_data['High'] - stock_data['Close'])) / (stock_data['High'] - stock_data['Low'])
    stock_data['CMF'] = (money_flow_volume * stock_data['Volume']).rolling(window=20).sum() / stock_data['Volume'].rolling(window=20).sum()


    # Rate of Change (ROC)
    stock_data['ROC'] = stock_data['Close'].pct_change(periods=10) * 100

    # Exponential Moving Average (EMA)
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

    # SMA indicators
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_100'] = stock_data['Close'].rolling(window=100).mean()

    # Drop rows with missing values after all calculations
    stock_data.dropna(inplace=True)

    return stock_data

# Function to create lag features
def create_lag_features(stock_data, lags=5):
    for lag in range(1, lags+1):
        stock_data[f'Close_Lag_{lag}'] = stock_data['Close'].shift(lag)
    stock_data.dropna(inplace=True)  # Drop rows with missing values
    return stock_data
#prepare data
def prepare_data(stock_data):
    # Define feature columns
    feature_columns = [
        'Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 
        'Bollinger_Low', 'MACD', 'Signal', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 
        'CCI', 'Momentum', 'MFI', 'CMF', 'ROC', 'EMA_20', 'EMA_50', 
        'SMA_20', 'SMA_50', 'SMA_100'
    ] + [f'Close_Lag_{i}' for i in range(1, 6)]

    # Filter data for features
    X = stock_data[feature_columns]

    # Normalize feature columns
    feature_scaler = MinMaxScaler()
    X_n = feature_scaler.fit_transform(X)

    y_high = stock_data['High'].values.reshape(-1, 1)
    y_low = stock_data['Low'].values.reshape(-1, 1)
    y_close = stock_data['Close'].values.reshape(-1, 1)
    y_open = stock_data['Open'].values.reshape(-1, 1)

    # Normalize target variables
    scaler_y_high = MinMaxScaler()
    scaler_y_low = MinMaxScaler()
    scaler_y_close = MinMaxScaler()
    scaler_y_open = MinMaxScaler()
    
    y_high_n = scaler_y_high.fit_transform(y_high)
    y_low_n = scaler_y_low.fit_transform(y_low)
    y_close_n = scaler_y_close.fit_transform(y_close)
    y_open_n = scaler_y_open.fit_transform(y_open)

    # Store scalers in session state for future use in predictions
    st.session_state.feature_scaler = feature_scaler
    st.session_state.scaler_y_high = scaler_y_high
    st.session_state.scaler_y_low = scaler_y_low
    st.session_state.scaler_y_close = scaler_y_close
    st.session_state.scaler_y_open = scaler_y_open

    # Return normalized features and targets
    return X_n, y_high_n, y_low_n, y_close_n, y_open_n

# Function to train XGBoost models for High, Low, and Close, Open prices
def train_xgboost_models(X_n, y_high_n, y_low_n, y_close_n, y_open_n):
    # Convert X to float32 for compatibility with XGBoost
    X_n = np.array(X_n)

    # Train models for High, Low, Close, and Open price predictions
    model_high = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.01, max_depth=100)
    model_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.01, max_depth=100)
    model_close = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.01, max_depth=100)
    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.01, max_depth=100)
    
    model_high.fit(X_n, y_high_n)
    model_low.fit(X_n, y_low_n)
    model_close.fit(X_n, y_close_n)
    model_open.fit(X_n, y_open_n)
    
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
    feature_scaler = st.session_state.feature_scaler
    st.session_state.stock_data = stock_data
    st.session_state.model_high = model_high
    st.session_state.model_low = model_low
    st.session_state.model_close = model_close
    st.session_state.model_open = model_open

# Allow the user to input the Open Price
if st.session_state.stock_data is not None:
    user_open_price = st.number_input("Enter the Open price", min_value=0.0, step=0.1)
    # "Predict Today" button to trigger prediction based on historical data
    if st.button('Predict Today'):
        # Get the Open price from the user
        if user_open_price > 0:  # Only proceed if a valid Open price is entered
            # Prepare the input data for prediction (using the last row as base)
            last_row = st.session_state.stock_data.iloc[-1:].copy()

            # Define the feature columns to be used for prediction
            features = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal', 
                        'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'Momentum', 'MFI', 'CMF', 'ROC', 'EMA_20', 'EMA_50', 
                        'SMA_20', 'SMA_50', 'SMA_100'] + [f'Close_Lag_{i}' for i in range(1, 6)]

            # Replace the Open price with the user input
            last_row['Open'] = user_open_price

            # Extract numerical values for the features
            feature_values = last_row[features].values.reshape(1, -1)

            # Normalize the features using the scaler from session state
            feature_scaler = st.session_state.feature_scaler
            feature_array = feature_scaler.transform(feature_values)

            # Make predictions with the trained models
            model_high = st.session_state.model_high
            model_low = st.session_state.model_low
            model_close = st.session_state.model_close
            model_open = st.session_state.model_open

            predicted_high_normalized = model_high.predict(feature_array)[0]
            predicted_low_normalized = model_low.predict(feature_array)[0]
            predicted_close_normalized = model_close.predict(feature_array)[0]
            predicted_open_normalized = model_open.predict(feature_array)[0]

            # Inverse transform the predictions using the fitted target scalers
            scaler_y_high = st.session_state.scaler_y_high
            scaler_y_low = st.session_state.scaler_y_low
            scaler_y_close = st.session_state.scaler_y_close
            scaler_y_open = st.session_state.scaler_y_open

            predicted_high = scaler_y_high.inverse_transform([[predicted_high_normalized]])[0][0]
            predicted_low = scaler_y_low.inverse_transform([[predicted_low_normalized]])[0][0]
            predicted_close = scaler_y_close.inverse_transform([[predicted_close_normalized]])[0][0]
            predicted_open = scaler_y_open.inverse_transform([[predicted_open_normalized]])[0][0]

            # Prepare data for the line chart
            last_30_days = st.session_state.stock_data[-30:]

            # Get the current date for labeling
            current_date = pd.Timestamp.today().strftime('%Y-%m-%d')

            predicted_data = pd.DataFrame({
                'Open': [predicted_open] * 2,  # Use the user-provided 'Open' value
                'High': [predicted_high] * 2,
                'Low': [predicted_low] * 2,
                'Close': [predicted_close] * 2,
                'Date': [last_30_days.index[-1], last_30_days.index[-1] + pd.Timedelta(days=1)]  # Current date and next day
            })

            # Optionally, display some more insights or predictions for other days if needed
            st.write(f"Here are the predicted prices for the {current_date} :")
            st.write(f"- Predicted Open: ₹{predicted_open:.2f}")
            st.write(f"- Predicted High: ₹{predicted_high:.2f}")
            st.write(f"- Predicted Low: ₹{predicted_low:.2f}")
            st.write(f"- Predicted Close: ₹{predicted_close:.2f}")

            # Provide a textual explanation of the chart
            st.write("""
            The line chart above shows the **Open**, **High**, **Low**, and **Close** prices for the last 7 days along with the predicted values for the next day.
            The predicted values are based on the model trained using historical stock data and technical indicators.
            Use this information for decision-making in the stock market.
            """)

            # Plot the historical and predicted data on the line chart
            st.line_chart(pd.concat([last_30_days[['Open', 'High', 'Low', 'Close']], predicted_data.set_index('Date')[['Open', 'High', 'Low', 'Close']]]))

        


    def predict_next_7_days(models, stock_data, lags=5):
            # Unpack the models
            model_high, model_low, model_close, model_open = models

            # Prepare a DataFrame to store future predictions
            future_predictions = []
            last_row = stock_data.iloc[-1:].copy()  # Start with the last available data

            for day in range(1,8):  # Predict for the next 7 days
                # Update lag features
                for lag in range(1, lags + 1):
                    last_row[f'Close_Lag_{lag}'] = stock_data['Close'].iloc[-lag]
                    last_row[f'Close_Lag_{lag}'] = stock_data['Open'].iloc[-lag]
                    last_row[f'Close_Lag_{lag}'] = stock_data['High'].iloc[-lag]
                    last_row[f'Close_Lag_{lag}'] = stock_data['Low'].iloc[-lag]

                        # Define the feature columns to be used for prediction
                features = ['Open', '5_day_MA', '10_day_MA', '50_day_MA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'Signal', 
                                'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'Momentum', 'MFI', 'CMF', 'ROC', 'EMA_20', 'EMA_50', 
                                'SMA_20', 'SMA_50', 'SMA_100'] + [f'Close_Lag_{i}' for i in range(1, 6)]
                    # Extract numerical values for the features
                last_row['Open'] = user_open_price    
                feature_values = last_row[features].values.reshape(1, -1)
                # Normalize the features using the scaler from session state
                feature_scaler = st.session_state.feature_scaler
                feature_array = feature_scaler.transform(feature_values)
                # Make predictions with the trained models
                model_high = st.session_state.model_high
                model_low = st.session_state.model_low
                model_close = st.session_state.model_close
                model_open = st.session_state.model_open
                predicted_high_normalized = model_high.predict(feature_array)[0]
                predicted_low_normalized = model_low.predict(feature_array)[0]
                predicted_close_normalized = model_close.predict(feature_array)[0]
                predicted_open_normalized = model_open.predict(feature_array)[0]
                # Inverse transform the predictions using the fitted target scalers
                scaler_y_high = st.session_state.scaler_y_high
                scaler_y_low = st.session_state.scaler_y_low
                scaler_y_close = st.session_state.scaler_y_close
                scaler_y_open = st.session_state.scaler_y_open
                predicted_high = scaler_y_high.inverse_transform([[predicted_high_normalized]])[0][0]
                predicted_low = scaler_y_low.inverse_transform([[predicted_low_normalized]])[0][0]
                predicted_close = scaler_y_close.inverse_transform([[predicted_close_normalized]])[0][0]
                predicted_open = scaler_y_open.inverse_transform([[predicted_open_normalized]])[0][0]
                # Append the predictions
                future_predictions.append({
                    'Date': last_row.index[0] + pd.Timedelta(days=day),
                    'Open':predicted_open,
                    'High': predicted_high,
                    'Low': predicted_low,
                    'Close': predicted_close
                })
                # Update the last_row for the next prediction
                last_row['Close'] = predicted_close
                last_row['Open'] = predicted_open  
                last_row['High'] = predicted_high  
                last_row['Low'] = predicted_low  
                stock_data = pd.concat([stock_data, pd.DataFrame({
                    'Close': [predicted_close], 'Open': [predicted_open], 
                    'High': [predicted_high], 'Low': [predicted_low]
                }, index=[last_row.index[0] + pd.Timedelta(days=day)])])
                 # stock_data = pd.concat([stock_data, pd.DataFrame({'Close': [predicted_close]}, index=[last_row.index[0] + pd.Timedelta(days=day)])])
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
    

