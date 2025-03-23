import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
df = pd.read_csv("profitable_stocks_new.csv")  # Replace with your actual file

# Drop unnecessary columns
features = [
    'open', 'high', 'low', 'close', 'volume', 'daily_change',
    'bullish_marubozu', 'bearish_marubozu', 'bullish_engulfing', 'bearish_engulfing',
    'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line',
    'support', 'resistance', 'stop_loss', 'target', 'rrr', 'valid_rrr'
]

# Prepare features (X) and target (y)
X = df[features]
y = df['profit_correct'].astype(int)  # Convert boolean to 0/1

# Handle missing values (if any)
X.fillna(X.median(), inplace=True)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {100 * accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, "stock_trading_model.pkl")
print("Model saved as stock_trading_model.pkl")
