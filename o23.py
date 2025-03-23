import yfinance as yf

symbol = "AAPL"  # Replace with any stock symbol
ticker = yf.Ticker(symbol)

# Get intraday (latest) data
latest_data = ticker.history(period="1d", interval="1m")  # 1-minute interval for today's data

print(latest_data.tail())  # Show the most recent rows
