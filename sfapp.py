import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# --- Page Settings ---
st.set_page_config("📊 Forecast App", layout="wide")

# --- Logo + Title ---
st.image("https://cdn-icons-png.flaticon.com/512/2103/2103627.png", width=60)
st.title("📈 Stock Forecasting Dashboard")

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
n_days = st.sidebar.slider("Forecast Horizon (Days)", 10, 60, 30)
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM", "SARIMA"])
plot_theme = st.sidebar.radio("Plot Theme", ["light", "dark"])

# --- Load Data from yFinance ---
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="2y")
    df = df[['Close']].dropna()
    df.index.name = 'Date'
    return df

df = load_data(symbol)
if df.empty:
    st.warning("No data found for symbol.")
    st.stop()

# --- Forecasting Functions ---
def forecast_arima(df):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=n_days)
    return pd.Series(pred, index=pd.date_range(df.index[-1], periods=n_days+1, freq='B')[1:])

def forecast_sarima(df):
    model = SARIMAX(df['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    pred = model_fit.forecast(n_days)
    return pd.Series(pred, index=pd.date_range(df.index[-1], periods=n_days+1, freq='B')[1:])

def forecast_prophet(df):
    data = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].tail(n_days)

def forecast_lstm(df):
    series = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    input_seq = scaled[-60:].reshape(1, 60, 1)
    preds = []
    for _ in range(n_days):
        next_val = model.predict(input_seq, verbose=0)[0][0]
        preds.append(next_val)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)
    inv_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return pd.Series(inv_preds, index=pd.date_range(df.index[-1], periods=n_days+1, freq='B')[1:])

# --- Run Forecast ---
if model_choice == "ARIMA":
    forecast = forecast_arima(df)
elif model_choice == "SARIMA":
    forecast = forecast_sarima(df)
elif model_choice == "Prophet":
    forecast = forecast_prophet(df)
else:
    forecast = forecast_lstm(df)

# --- Combine Actual + Forecast ---
recent = df['Close'].iloc[-30:]
combined = pd.concat([recent, forecast])
max_date = combined.idxmax()
min_date = combined.idxmin()

# --- Plot Theme Setting ---
plot_theme_dict = {
    "light": "plotly_white",
    "dark": "plotly_dark"
}

# --- Plot Forecast ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode='lines', name="Recent Prices"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name="Forecast"))
fig.add_trace(go.Scatter(x=[max_date], y=[combined.max()], mode='markers+text', name="Max",
                         marker=dict(size=10, color='green'), text=["Max"], textposition="top center"))
fig.add_trace(go.Scatter(x=[min_date], y=[combined.min()], mode='markers+text', name="Min",
                         marker=dict(size=10, color='red'), text=["Min"], textposition="bottom center"))
fig.update_layout(title=f"{symbol} Forecast ({model_choice})", xaxis_title="Date", yaxis_title="Price",
                  template=plot_theme_dict[plot_theme])
st.plotly_chart(fig, use_container_width=True)

# --- Quick Metrics ---
st.subheader("📊 Forecast Summary")
st.metric("📈 Max Forecasted Price", f"${forecast.max():.2f}")
st.metric("📉 Min Forecasted Price", f"${forecast.min():.2f}")

# --- Download Table ---
forecast_df = forecast.reset_index()
forecast_df.columns = ['Date', 'Forecasted Price']
st.dataframe(forecast_df)
csv = forecast_df.to_csv(index=False).encode()
st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown("📘 **Made with ❤️ by Mayakshi · Real-time Stock Forecasting App · Powered by Streamlit**")