import pandas as pd
import numpy as np
import yfinance as yf
import joblib  # Using joblib for model loading

# Load trained model
model = joblib.load("stock_trading_model.pkl")


def get_ticker_daily_price(symbol, period="1mo"):
    """Fetches historical stock data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"❌ No data found for {symbol}!")

    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                         'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    data['symbol'] = symbol
    return data


def detect_candlestick_patterns(data):
    """Detects candlestick patterns (Marubozu, Engulfing)."""
    data['bullish_marubozu'] = (data['close'] > data['open']) & ((data['close'] - data['low']) > 0.95 * (data['high'] - data['low']))
    data['bearish_marubozu'] = (data['close'] < data['open']) & ((data['high'] - data['open']) > 0.95 * (data['high'] - data['low']))

    data['bullish_engulfing'] = (data['close'] > data['open']) & (data['open'].shift(1) > data['close'].shift(1)) & (data['close'] > data['open'].shift(1))
    data['bearish_engulfing'] = (data['close'] < data['open']) & (data['open'].shift(1) < data['close'].shift(1)) & (data['close'] < data['open'].shift(1))

    return data


def calculate_trends(data):
    """Calculates RSI, MACD, and identifies trends."""
    delta = data['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))

    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data


def calculate_support_resistance(data):
    """Computes support and resistance levels."""
    data['support'] = data['low'].rolling(window=10, min_periods=1).min()
    data['resistance'] = data['high'].rolling(window=10, min_periods=1).max()
    return data


def calculate_rrr(data):
    """Calculates Risk-Reward Ratio (RRR) and defines stop loss & target."""
    data['stop_loss'] = np.where(data['close'] > data['open'], data['support'] * 0.98, data['resistance'] * 1.02)
    data['target'] = np.where(data['close'] > data['open'], data['resistance'], data['support'])

    stop_loss_diff = data['close'] - data['stop_loss']
    stop_loss_diff = np.where(stop_loss_diff == 0, np.nan, stop_loss_diff)

    data['rrr'] = (data['target'] - data['close']) / stop_loss_diff
    data['valid_rrr'] = data['rrr'].fillna(0) > 1.5
    return data


def prepare_input_data(row):
    """Prepares feature row for model prediction."""
    feature_data = {
        'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
        'volume': row['volume'], 'daily_change': row['close'] - row['open'],
        'bullish_marubozu': row['bullish_marubozu'], 'bearish_marubozu': row['bearish_marubozu'],
        'bullish_engulfing': row['bullish_engulfing'], 'bearish_engulfing': row['bearish_engulfing'],
        'RSI': row['RSI'], 'EMA_12': row['EMA_12'], 'EMA_26': row['EMA_26'],
        'MACD': row['MACD'], 'Signal_Line': row['Signal_Line'],
        'support': row['support'], 'resistance': row['resistance'],
        'stop_loss': row['stop_loss'], 'target': row['target'],
        'rrr': row['rrr'], 'valid_rrr': row['valid_rrr']
    }

    return pd.DataFrame([feature_data])


def predict_profitability(stock_symbol):
    """Fetches stock data, computes indicators, and predicts profitability for all available dates."""
    print(f"📊 Fetching data for {stock_symbol}...")

    # Fetch & Process Stock Data
    data = get_ticker_daily_price(stock_symbol, period="1mo")
    data = detect_candlestick_patterns(data)
    data = calculate_trends(data)
    data = calculate_support_resistance(data)
    data = calculate_rrr(data)

    # Initialize result list
    predictions = []

    for date in data['timestamp']:
        row = data[data['timestamp'] == date].iloc[-1]  # Get row for the date
        input_data = prepare_input_data(row)

        # Ensure Features Match Model Training
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'daily_change',
            'bullish_marubozu', 'bearish_marubozu', 'bullish_engulfing', 'bearish_engulfing',
            'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line',
            'support', 'resistance', 'stop_loss', 'target', 'rrr', 'valid_rrr'
        ]
        X_test = input_data[feature_columns]
        X_test.fillna(X_test.median(), inplace=True)

        # Make Prediction
        prediction = model.predict(X_test)[0]

        # Store result
        predictions.append({"timestamp": date, "predicted_profitability": prediction})

    # Convert results to DataFrame
    result_df = pd.DataFrame(predictions)

    # Print predictions
    print("\n✅ **Predicted Profitability for All Dates:**")
    print(result_df)

    return result_df


# Example Usage
stock_symbol = "SBIN.NS"  # Change to any stock symbol

try:
    result = predict_profitability(stock_symbol)
except ValueError as e:
    print(e)
