import pandas as pd
import numpy as np
import os

# Define Data Directory
DATA_DIR = "../stock_data"
os.makedirs(DATA_DIR, exist_ok=True)

# List of stock files 'HDFC.NSE.csv', 'SBI.NSE.csv', 'INFY.NSE.csv', 'NIFTY_50.csv'
stock_files = ['HDFC.NSE.csv', 'SBI.NSE.csv', 'INFY.NSE.csv', 'NIFTY_50.csv']


def load_existing_data(file_path):
    """Load existing stock data from CSV file."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found! Skipping...")
        return None

    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}")
    return data


def clean_stock_data(data):
    """Cleans stock data by fixing column names and converting numeric columns."""
    # Strip column names of extra spaces
    data.columns = data.columns.str.strip()

    # Convert numeric columns (including volume)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in data.columns:  # Ensure column exists before conversion
            data[col] = data[col].astype(str).str.replace(',', '', regex=True).astype(float)

    return data



def preprocess_data(data):
    """Clean and preprocess the stock data."""
    data = data.copy()
    data.columns = data.columns.str.strip().str.lower()

    required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data.dropna(subset=['timestamp'], inplace=True)
    data.sort_values(by='timestamp', inplace=True)
    data.drop_duplicates(subset=['timestamp'], inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.interpolate(method='linear', inplace=True)

    data['daily_change'] = data['close'] - data['open']
    return data


def detect_candlestick_patterns(data):
    """Detect candlestick patterns."""
    data['bullish_marubozu'] = (data['close'] > data['open']) & (
                (data['close'] - data['low']) > 0.95 * (data['high'] - data['low']))
    data['bearish_marubozu'] = (data['close'] < data['open']) & (
                (data['high'] - data['open']) > 0.95 * (data['high'] - data['low']))

    data['bullish_engulfing'] = (data['close'] > data['open']) & (data['open'].shift(1) > data['close'].shift(1)) & (
                data['close'] > data['open'].shift(1))
    data['bearish_engulfing'] = (data['close'] < data['open']) & (data['open'].shift(1) < data['close'].shift(1)) & (
                data['close'] < data['open'].shift(1))

    return data


def calculate_trends(data):
    """Calculate RSI, MACD, and classify trends."""
    delta = data['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    conditions = [
        (data['RSI'] > 70) & (data['MACD'] > data['Signal_Line']),
        (data['RSI'] < 30) & (data['MACD'] < data['Signal_Line']),
        (data['MACD'] > data['Signal_Line']),
        (data['MACD'] < data['Signal_Line'])
    ]
    choices = ['Strong Uptrend', 'Strong Downtrend', 'Uptrend', 'Downtrend']
    data['trend'] = np.select(conditions, choices, default='Neutral')

    return data


def calculate_support_resistance(data):
    """Calculate support and resistance levels."""
    data['support'] = data['low'].rolling(window=10, min_periods=1).min()
    data['resistance'] = data['high'].rolling(window=10, min_periods=1).max()
    return data


def calculate_rrr(data):
    """Calculate risk-reward ratio."""
    data['stop_loss'] = np.where(data['close'] > data['open'], data['support'] * 0.98, data['resistance'] * 1.02)
    data['target'] = np.where(data['close'] > data['open'], data['resistance'], data['support'])

    stop_loss_diff = data['close'] - data['stop_loss']
    stop_loss_diff = np.where(stop_loss_diff == 0, np.nan, stop_loss_diff)

    data['rrr'] = (data['target'] - data['close']) / stop_loss_diff
    data['valid_rrr'] = data['rrr'].fillna(0) > 1.5
    return data


def check_profit_correctness(data):
    """Check if the target or stop-loss was hit first."""
    data['profit_correct'] = False
    for i in range(len(data) - 5):
        future_data = data.iloc[i + 1: i + 6]
        stop_hit = (future_data['low'] <= data.loc[i, 'stop_loss']).any()
        target_hit = (future_data['high'] >= data.loc[i, 'target']).any()
        data.at[i, 'profit_correct'] = target_hit and not stop_hit
    return data


def evaluate_stock(data):
    """Evaluate stocks based on patterns, volume, trend, and RRR."""
    data['valid_stock'] = (
            data[['bullish_marubozu', 'bearish_marubozu', 'bullish_engulfing', 'bearish_engulfing']].any(axis=1) |
            data['valid_rrr']
    )
    data['expected_profit'] = np.where(data['valid_stock'],
                                       ((data['resistance'] - data['close']) / data['close']) * 100, 0)
    data['expected_loss'] = np.where(data['valid_stock'], ((data['close'] - data['stop_loss']) / data['close']) * 100,
                                     0)
    return data


def process_all_stocks(stock_files):
    """Process multiple stock files and combine profitable stocks."""
    all_profitable_stocks = []

    for stock_file in stock_files:
        file_path = os.path.join(DATA_DIR, stock_file)
        try:
            data = load_existing_data(file_path)
            data = clean_stock_data(data)
            data = preprocess_data(data)
            data = detect_candlestick_patterns(data)
            data = calculate_trends(data)
            data = calculate_support_resistance(data)
            data = calculate_rrr(data)
            data = evaluate_stock(data)
            data = check_profit_correctness(data)

            profitable_stocks = data[data['expected_profit'] > 0]
            all_profitable_stocks.append(profitable_stocks)
        except Exception as e:
            print(f"Error processing {stock_file}: {e}")

    if all_profitable_stocks:

        final_df = pd.concat(all_profitable_stocks, ignore_index=True)
        final_df.to_csv("profitable_stocks.csv", index=False)
        print(final_df.columns)
        print("Saved profitable stocks to profitable_stocks_new.csv")
    else:
        print("No profitable stocks found.")


if __name__ == "__main__":
    process_all_stocks(stock_files)
