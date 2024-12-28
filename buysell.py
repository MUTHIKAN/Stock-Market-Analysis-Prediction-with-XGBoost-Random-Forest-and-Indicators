import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# Function to fetch live stock data
def fetch_live_data(ticker, interval="5m", period="1d"):
    try:
        data = yf.download(tickers=ticker, interval=interval, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to detect candlestick patterns
# Function to detect candlestick patterns row-by-row
def detect_patterns(data):
    def check_pattern(row):
        try:
            # Explicitly convert to float
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])

            # Doji Pattern
            if abs(close_price - open_price) <= 0.1 * (high_price - low_price):
                return "Doji"

            # Hammer Pattern
            elif close_price > open_price and (close_price - open_price) < 0.2 * (high_price - low_price) and (open_price - low_price) >= 2 * (close_price - open_price):
                return "Hammer"

            # Shooting Star Pattern
            elif open_price > close_price and (open_price - close_price) < 0.2 * (high_price - low_price) and (high_price - open_price) >= 2 * (open_price - close_price):
                return "Shooting Star"

            # No Pattern
            else:
                return "No Pattern"
        except Exception as e:
            return f"Error: {e}"

    # Apply the check_pattern function to each row
    data["Pattern"] = data.apply(check_pattern, axis=1)
    return data


    # Apply the check_pattern function to each row in the DataFrame
    data["Pattern"] = data.apply(check_pattern, axis=1)
    return data


# Streamlit App
def main():
    st.title("Indian Stock Market Candlestick Pattern Detection")
    
    # Sidebar for user input
    st.sidebar.header("Stock Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.BO, RELIANCE.BO):", "TCS.BO")
    interval = st.sidebar.selectbox("Select Interval:", ["1m", "5m", "15m"], index=0)
    period = st.sidebar.selectbox("Select Period:", ["1d", "5d"], index=0)

    # Fetch and display stock data
    data = fetch_live_data(ticker, interval, period)
    if data is not None:
        st.subheader(f"Live Data for {ticker}")
        st.write(data.tail())
        
        # Detect and display patterns
        patterns = detect_patterns(data)
        patterns_df = pd.DataFrame(patterns, columns=["Time", "Pattern"])
        st.subheader("Detected Candlestick Patterns")
        st.write(patterns_df)

        # Plot candlestick chart
        st.subheader("Candlestick Chart")
        mpf_style = mpf.make_mpf_style(base_mpf_style="yahoo", rc={"font.size": 10})
        fig, ax = mpf.plot(data, type="candle", style=mpf_style, returnfig=True, title=ticker, volume=True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
