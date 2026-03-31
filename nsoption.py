import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import yfinance as yf

st.set_page_config(layout="wide")

# -------------------------------
# Custom Styling (Modern UI)
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
.metric-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Black-Scholes Model
# -------------------------------
def black_scholes(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)

    d1 = (np.log(S / K) + (r + sigma**2 / 2)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_greeks(S, K, T, r, sigma, option_type):
    if T <= 0:
        return 0, 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + sigma**2 / 2)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "Call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "Call" else -d2)) / 100

    return delta, gamma, theta, vega, rho


# -------------------------------
# Fetch Prices
# -------------------------------
def fetch_live_prices():
    nifty = yf.Ticker("^NSEI")
    sensex = yf.Ticker("^BSESN")

    return (
        nifty.history(period="1d")["Close"].iloc[-1],
        sensex.history(period="1d")["Close"].iloc[-1]
    )


# -------------------------------
# Header
# -------------------------------
def header_box(nifty, sensex):
    today = datetime.now().strftime("%d %b %Y")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric-box'>📅<br>{today}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'>NIFTY<br>{nifty:.2f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'>SENSEX<br>{sensex:.2f}</div>", unsafe_allow_html=True)


# -------------------------------
# Init State
# -------------------------------
if "nifty" not in st.session_state:
    st.session_state["nifty"] = 22000.0
if "sensex" not in st.session_state:
    st.session_state["sensex"] = 73000.0

if st.button("🔄 Fetch Live Prices"):
    n, s = fetch_live_prices()
    st.session_state["nifty"] = n
    st.session_state["sensex"] = s

nifty_price = st.session_state["nifty"]
sensex_price = st.session_state["sensex"]

header_box(nifty_price, sensex_price)

st.title("📊 Options Strategy Dashboard")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")
r = st.sidebar.number_input("Risk-Free Rate (%)", value=6.5) / 100
iv_adjust = st.sidebar.slider("IV Adjustment (%)", -50, 50, 0) / 100
time_shift = st.sidebar.slider("Days Forward", 0, 30, 0)

# -------------------------------
# Strategy Builder
# -------------------------------
legs = []

st.subheader("🧩 Strategy Builder")

for i in range(6):
    with st.container():
        col0, col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1,1])

        enabled = col0.checkbox("", key=f"enable{i}", value=(i==0))

        underlying = col1.selectbox("Underlying", ["NIFTY", "SENSEX"], key=f"u{i}")
        strike = col2.number_input("Strike", value=22000 if underlying=="NIFTY" else 73000, key=f"k{i}")
        premium = col3.number_input("Premium", value=100.0, key=f"p{i}")
        position = col4.selectbox("Pos", ["Long", "Short"], key=f"pos{i}")
        option_type = col5.selectbox("Type", ["Call", "Put"], key=f"type{i}")
        expiry = col6.date_input("Expiry", datetime.today(), key=f"exp{i}")

        if enabled:
            legs.append({
                "underlying": underlying,
                "strike": strike,
                "premium": premium,
                "position": position,
                "type": option_type,
                "expiry": expiry
            })

# -------------------------------
# Payoff Engine
# -------------------------------
def calculate_payoff(legs):
    pct_range = np.linspace(-0.1, 0.1, 120)
    total = np.zeros_like(pct_range)
    individual = []

    for leg in legs:
        S0 = nifty_price if leg["underlying"] == "NIFTY" else sensex_price
        S_range = S0 * (1 + pct_range)

        T_days = (leg["expiry"] - datetime.today().date()).days - time_shift
        T = max(T_days / 365, 0.0001)

        sigma = 0.2 + iv_adjust

        payoff = []

        for S in S_range:
            price = black_scholes(S, leg["strike"], T, r, sigma, leg["type"])
            pnl = price - leg["premium"]
            if leg["position"] == "Short":
                pnl = -pnl
            payoff.append(pnl)

        payoff = np.array(payoff)
        total += payoff
        individual.append(payoff)

    return pct_range * 100, total, individual


# -------------------------------
# Plot
# -------------------------------
if st.button("🚀 Calculate Payoff"):
    if not legs:
        st.warning("Select at least one active leg.")
    else:
        x, total, individual = calculate_payoff(legs)

        df = pd.DataFrame({"% Change": x, "Total": total})

        for idx, leg_payoff in enumerate(individual):
            df[f"Leg {idx+1}"] = leg_payoff

        st.subheader("📈 Payoff Visualization")

        selected = st.multiselect(
            "Select legs to display",
            options=["Total"] + [f"Leg {i+1}" for i in range(len(individual))],
            default=["Total"]
        )

        st.line_chart(df.set_index("% Change")[selected])

        st.subheader("📊 Greeks Summary")

        greeks = []
        for leg in legs:
            S0 = nifty_price if leg["underlying"] == "NIFTY" else sensex_price
            T = max(((leg["expiry"] - datetime.today().date()).days - time_shift) / 365, 0.0001)
            sigma = 0.2 + iv_adjust
            greeks.append(calculate_greeks(S0, leg["strike"], T, r, sigma, leg["type"]))

        greek_df = pd.DataFrame(greeks, columns=["Delta", "Gamma", "Theta", "Vega", "Rho"])
        st.dataframe(greek_df, use_container_width=True)
