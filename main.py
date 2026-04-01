# ==============================
# 📦 IMPORT LIBRARIES
# ==============================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==============================
# 📥 USER INPUT
# ==============================
stock_symbol = input("Enter stock symbol (AAPL, TSLA, MSFT): ").upper()


# ==============================
# 📊 FETCH DATA
# ==============================
print(f"\nFetching data for {stock_symbol}...")

stock = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")

if stock.empty:
    print("❌ Invalid stock symbol or no data found.")
    exit()

print("\nSample Data:")
print(stock.head())


# ==============================
# 🧹 FEATURE ENGINEERING
# ==============================
stock['Price_Change'] = stock['Close'] - stock['Open']
stock['High_Low_Diff'] = stock['High'] - stock['Low']

# Target (next day's close)
stock['Target'] = stock['Close'].shift(-1)

# Drop missing values
stock.dropna(inplace=True)


# ==============================
# 🎯 FEATURES & LABEL
# ==============================
X = stock[['Open', 'High', 'Low', 'Volume', 'Price_Change', 'High_Low_Diff']]
y = stock['Target']


# ==============================
# ✂️ TRAIN TEST SPLIT (TIME SERIES SAFE)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# ==============================
# 🤖 MODEL 1: LINEAR REGRESSION
# ==============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


# ==============================
# 🌲 MODEL 2: RANDOM FOREST
# ==============================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


# ==============================
# 📏 MODEL EVALUATION
# ==============================
print("\n📊 Model Performance:")

print("\nLinear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", mean_squared_error(y_test, lr_pred, squared=False))

print("\nRandom Forest:")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", mean_squared_error(y_test, rf_pred, squared=False))


# ==============================
# 📈 VISUALIZATION
# ==============================
plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual Price", linewidth=2)
plt.plot(rf_pred, label="RF Prediction", linestyle="--")
plt.plot(lr_pred, label="LR Prediction", linestyle="--")

plt.legend()
plt.title(f"{stock_symbol} Stock Price Prediction (Next Day Close)")
plt.xlabel("Time")
plt.ylabel("Price")

plt.grid()
plt.show()


# ==============================
# 🔮 PREDICT NEXT DAY PRICE
# ==============================
latest_data = X.iloc[-1].values.reshape(1, -1)
tomorrow_price = rf_model.predict(latest_data)

print(f"\n📌 Predicted Next Day Closing Price for {stock_symbol}: {tomorrow_price[0]:.2f}")


# ==============================
# 💾 SAVE MODEL
# ==============================
model_filename = f"{stock_symbol}_stock_model.pkl"
joblib.dump(rf_model, model_filename)

print(f"\n✅ Model saved as {model_filename}")