import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objs as go

st.set_page_config(page_title="Stock Forecasting", layout="wide")

# Theme toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    plt.style.use("dark_background")
    st.markdown("<style>body{background-color:#111;color:white;}</style>", unsafe_allow_html=True)
else:
    plt.style.use("default")

# Header and logo
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331943.png", width=80)
st.title("📈 Stock Forecasting App")
st.markdown("Enter a stock symbol (e.g., `AAPL`, `GOOGL`, `TSLA`) to see predictions using different models.")

# Input
ticker = st.text_input("Stock Symbol", "AAPL").upper()
n_days = st.slider("Forecast Days", 7, 90, 30)

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    df.reset_index(inplace=True)
    return df

df = load_data(ticker)
st.write("Last 5 records:")
st.dataframe(df.tail())

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Closing Price"))
fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# Tabs for models
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prophet", "🔁 ARIMA", "📊 SARIMA", "🧠 LSTM"])

# Prophet Model
with tab1:
    prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=n_days)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    st.pyplot(fig1)
    st.metric("🔺 Max Forecasted Price", f"${forecast['yhat'].max():.2f}")
    st.metric("🔻 Min Forecasted Price", f"${forecast['yhat'].min():.2f}")

# ARIMA
with tab2:
    arima_df = df.set_index("Date")["Close"]
    model = ARIMA(arima_df, order=(5,1,0))
    model_fit = model.fit()
    forecast_arima = model_fit.forecast(steps=n_days)
    st.line_chart(forecast_arima)
    st.metric("🔺 Max Forecasted Price", f"${forecast_arima.max():.2f}")
    st.metric("🔻 Min Forecasted Price", f"${forecast_arima.min():.2f}")

# SARIMA
with tab3:
    sarima_model = SARIMAX(arima_df, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_result = sarima_model.fit(disp=False)
    forecast_sarima = sarima_result.forecast(steps=n_days)
    st.line_chart(forecast_sarima)
    st.metric("🔺 Max Forecasted Price", f"${forecast_sarima.max():.2f}")
    st.metric("🔻 Min Forecasted Price", f"${forecast_sarima.min():.2f}")

# LSTM
with tab4:
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    training_data_len = int(np.ceil(len(dataset) * .8))

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    st.line_chart(valid[['Close', 'Predictions']])
    st.metric("🔺 Max Forecasted Price", f"${valid['Predictions'].max():.2f}")
    st.metric("🔻 Min Forecasted Price", f"${valid['Predictions'].min():.2f}")
    # --- Download Table ---
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Date', 'Forecasted Price']
    st.dataframe(forecast_df)
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

    # --- Footer ---
    st.markdown("---")
    st.markdown("📘 *Developed by Mayakshi · Real-time Stock Forecasting App · Powered by Streamlit*")
