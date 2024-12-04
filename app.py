import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit header
st.header('Stock Market Predictor Using LSTM')

# Stock Symbol input from user
stock = st.text_input('Enter Stock Symbol', 'TSLA')

# Define the start and end dates
start = '2012-01-01'
end = '2022-12-31'

# Download stock data from Yahoo Finance
data = yf.download(stock, start=start, end=end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Split the data into training and testing sets (80% train, 20% test)
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): len(data)])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Prepare the test data by concatenating last 100 days of train data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)

# Prepare data for prediction (using 100 previous days)
x_train, y_train = [], []
for i in range(100, data_train_scaled.shape[0]):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the LSTM Model with improved configurations
model = Sequential()

# LSTM Layer with 100 neurons
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# LSTM Layer with 100 neurons
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Dense output layer
model.add(Dense(units=1))  # Single output for prediction

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model with more epochs
model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

# Prepare the test data for prediction (using the last 100 days from the test data)
x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions using the trained model
predicted_prices = model.predict(x_test)

# Rescale predictions back to the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate error metrics
mae = mean_absolute_error(y_actual, predicted_prices)
mse = mean_squared_error(y_actual, predicted_prices)
r2 = r2_score(y_actual, predicted_prices)

# Display the error metrics
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"R-Squared (R²): {r2:.4f}")

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(y_actual, predicted_prices)
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Plotting the results
st.subheader('Original Price vs Predicted Price')
fig = plt.figure(figsize=(8,6))
plt.plot(y_actual, 'g', label='Original Price')
plt.plot(predicted_prices, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock} Price Prediction vs Actual')
plt.legend()
st.pyplot(fig)

# Displaying moving averages (50, 100, 200 days)
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label="MA 50")
plt.plot(data['Close'], 'g', label="Price")
plt.title(f'{stock} Price vs MA50')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label="MA 50")
plt.plot(ma_100_days, 'b', label="MA 100")
plt.plot(data['Close'], 'g', label="Price")
plt.title(f'{stock} Price vs MA50 vs MA100')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label="MA 100")
plt.plot(ma_200_days, 'b', label="MA 200")
plt.plot(data['Close'], 'g', label="Price")
plt.title(f'{stock} Price vs MA100 vs MA200')
plt.legend()
st.pyplot(fig3)

# Conclusion section
st.subheader('Model Accuracy & Comparison')
st.write(f"The model's accuracy is based on the following metrics:")
st.write(f"MAE: Mean Absolute Error (Lower is better)")
st.write(f"MSE: Mean Squared Error (Lower is better)")
st.write(f"R²: R-squared (Higher is better)")
st.write(f"MAPE: Mean Absolute Percentage Error (Lower is better)")
