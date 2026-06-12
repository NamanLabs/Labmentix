import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
from datetime import timedelta

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────────
st.title("📈 Tesla (TSLA) Stock Price Predictor")
st.markdown("**Deep Learning Model — SimpleRNN vs LSTM**")
st.markdown("---")

# ── Load Model & Scaler ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    import tensorflow as tf
    model = tf.keras.models.load_model("tsla_best_lstm.keras")
    with open("tsla_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

# ── Check files exist ─────────────────────────────────────────────────
missing = [f for f in ["tsla_best_lstm.keras", "tsla_scaler.pkl", "TSLA.csv"]
           if not os.path.exists(f)]

if missing:
    st.error(f"❌ Missing files: {', '.join(missing)}")
    st.info(
        "**Steps to fix:**\n"
        "1. Run the notebook end-to-end first\n"
        "2. Copy `tsla_best_lstm.keras`, `tsla_scaler.pkl`, and `TSLA.csv` "
        "into the same folder as `app.py`\n"
        "3. Re-run `streamlit run app.py`"
    )
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────
try:
    model, scaler = load_assets()
    df = load_data()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

LOOKBACK = 50

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Forecast Settings")
forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[1, 5, 10],
    index=1,
    format_func=lambda x: f"{x}-Day Ahead"
)
show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
chart_theme = st.sidebar.selectbox("Chart Color Theme", ["Default", "Dark", "Pastel"])

color_map = {
    "Default": {"actual": "steelblue",  "pred": "tomato",   "ma50": "orange", "ma200": "red"},
    "Dark":    {"actual": "cyan",        "pred": "magenta",  "ma50": "yellow", "ma200": "lime"},
    "Pastel":  {"actual": "cornflowerblue", "pred": "salmon","ma50": "peachpuff","ma200": "lightcoral"},
}
colors = color_map[chart_theme]

# ── Metrics Row ───────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
last_price  = df["Adj Close"].iloc[-1]
prev_price  = df["Adj Close"].iloc[-2]
price_delta = last_price - prev_price
pct_change  = (price_delta / prev_price) * 100

col1.metric("Last Close Price",   f"${last_price:.2f}",  f"{price_delta:+.2f} ({pct_change:+.2f}%)")
col2.metric("Dataset Start",      str(df.index.min().date()))
col3.metric("Dataset End",        str(df.index.max().date()))
col4.metric("Total Trading Days", f"{len(df):,}")

st.markdown("---")

# ── Forecast Function ─────────────────────────────────────────────────
def recursive_forecast(model, scaler, scaled_data, n_steps, lookback):
    seq = scaled_data[-lookback:].copy()
    preds = []
    for _ in range(n_steps):
        inp  = seq.reshape(1, lookback, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        seq = np.append(seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ── Scale data ────────────────────────────────────────────────────────
scaled_data = scaler.transform(df[["Adj Close"]].values)

# ── Run forecast ──────────────────────────────────────────────────────
with st.spinner(f"Generating {forecast_days}-day forecast..."):
    forecast = recursive_forecast(model, scaler, scaled_data, forecast_days, LOOKBACK)

future_dates = pd.date_range(
    start=df.index[-1] + timedelta(days=1),
    periods=forecast_days,
    freq="B"
)

# ── CHART 1: Historical Price + Forecast ─────────────────────────────
st.subheader("📊 Historical Price & Forecast")

n_history = st.slider("Days of historical data to show", 30, 500, 120, step=10)

hist_df = df["Adj Close"].iloc[-n_history:]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(hist_df.index, hist_df.values, color=colors["actual"], lw=1.5, label="Historical Price")

if show_technical:
    ma50  = df["Adj Close"].rolling(50).mean().iloc[-n_history:]
    ma200 = df["Adj Close"].rolling(200).mean().iloc[-n_history:]
    ax.plot(hist_df.index, ma50,  color=colors["ma50"],  lw=1.2, linestyle="--", label="MA-50")
    ax.plot(hist_df.index, ma200, color=colors["ma200"], lw=1.2, linestyle="--", label="MA-200")

ax.axvline(df.index[-1], color="gray", lw=1.0, linestyle="--", alpha=0.7, label="Forecast Start")
ax.plot(future_dates, forecast, "o-", color=colors["pred"], lw=2.0,
        markersize=7, label=f"{forecast_days}-Day Forecast")

for i, (d, p) in enumerate(zip(future_dates, forecast)):
    ax.annotate(f"${p:.2f}", (d, p), textcoords="offset points",
                xytext=(5, 6), fontsize=8.5, color=colors["pred"])

ax.set_title(f"TSLA — Last {n_history} Days + {forecast_days}-Day Forecast", fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Forecast Table ────────────────────────────────────────────────────
st.subheader(f"📅 {forecast_days}-Day Price Forecast")

forecast_df = pd.DataFrame({
    "Date":            future_dates.strftime("%Y-%m-%d"),
    "Predicted Price": [f"${p:.2f}" for p in forecast],
    "Change from Last Close": [f"{p - last_price:+.2f} ({(p - last_price)/last_price*100:+.2f}%)"
                                for p in forecast]
})
forecast_df.index = range(1, len(forecast_df) + 1)
forecast_df.index.name = "Day"
st.dataframe(forecast_df, use_container_width=True)

# ── CHART 2: Bollinger Bands ──────────────────────────────────────────
if show_technical:
    st.subheader("📉 Bollinger Bands (Last 200 Days)")

    bb_df = df["Adj Close"].iloc[-200:]
    bb_mid   = bb_df.rolling(20).mean()
    bb_upper = bb_mid + 2 * bb_df.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_df.rolling(20).std()

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.plot(bb_df.index, bb_df.values, color=colors["actual"], lw=1.2, label="Adj Close")
    ax2.plot(bb_df.index, bb_mid,   color="orange", lw=1.0, linestyle="--", label="MA-20")
    ax2.plot(bb_df.index, bb_upper, color="green",  lw=1.0, linestyle="--", label="BB Upper")
    ax2.plot(bb_df.index, bb_lower, color="red",    lw=1.0, linestyle="--", label="BB Lower")
    ax2.fill_between(bb_df.index, bb_upper, bb_lower, alpha=0.08, color="gray")
    ax2.set_title("Bollinger Bands", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── CHART 3: Volume ───────────────────────────────────────────────────
    st.subheader("📦 Trading Volume (Last 200 Days)")

    vol_df = df["Volume"].iloc[-200:]
    fig3, ax3 = plt.subplots(figsize=(13, 3))
    ax3.bar(vol_df.index, vol_df.values, width=1, color="teal", alpha=0.5)
    ax3.plot(vol_df.index, vol_df.rolling(20).mean(), color="darkred", lw=1.5, label="20-day Avg")
    ax3.set_title("Trading Volume", fontsize=12, fontweight="bold")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Summary Stats ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Dataset Summary")
col_a, col_b = st.columns(2)
with col_a:
    st.dataframe(df[["Open","High","Low","Adj Close","Volume"]].describe().round(2),
                 use_container_width=True)
with col_b:
    st.markdown("### Model Info")
    st.info(
        f"**Architecture:** Stacked LSTM (Tuned)\n\n"
        f"**Lookback Window:** {LOOKBACK} days\n\n"
        f"**Training Data:** 80% chronological split\n\n"
        f"**Optimizer:** Adam\n\n"
        f"**Loss Function:** Mean Squared Error"
    )

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit | TSLA Data: 2010–2020")
