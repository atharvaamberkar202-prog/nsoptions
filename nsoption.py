import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import yfinance as yf

st.set_page_config(layout="wide")

# -------------------------------
# Black-Scholes Model
# -------------------------------
def black_scholes(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def calculate_greeks(S, K, T, r, sigma, option_type):
    if T <= 0:
        return 0, 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "Call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "Call" else -d2)) / 100

    return delta, gamma, theta, vega, rho


# -------------------------------
# Fetch Live Prices
# -------------------------------
def fetch_live_prices():
    nifty = yf.Ticker("^NSEI")
    sensex = yf.Ticker("^BSESN")

    nifty_price = nifty.history(period="1d")["Close"].iloc[-1]
    sensex_price = sensex.history(period="1d")["Close"].iloc[-1]

    return nifty_price, sensex_price


# -------------------------------
# UI Layout
# -------------------------------
st.title("Options Payoff Dashboard (NIFTY & SENSEX)")

if st.button("Fetch Live Prices"):
    nifty_price, sensex_price = fetch_live_prices()
    st.session_state["nifty"] = nifty_price
    st.session_state["sensex"] = sensex_price

nifty_price = st.session_state.get("nifty", 22000)
sensex_price = st.session_state.get("sensex", 73000)

st.sidebar.header("Market Inputs")
r = st.sidebar.number_input("Risk-Free Rate (%)", value=6.5) / 100
iv_adjust = st.sidebar.slider("IV Adjustment (%)", -50, 50, 0) / 100
time_shift = st.sidebar.slider("Days Forward", 0, 30, 0)

# -------------------------------
# Multi-leg Input
# -------------------------------
legs = []

st.subheader("Strategy Builder (Max 6 Legs)")

for i in range(6):
    with st.expander(f"Leg {i+1}"):
        col1, col2, col3 = st.columns(3)

        underlying = col1.selectbox("Underlying", ["NIFTY", "SENSEX"], key=f"u{i}")
        strike = col2.number_input("Strike", value=22000 if underlying=="NIFTY" else 73000, key=f"k{i}")
        premium = col3.number_input("Premium", value=100.0, key=f"p{i}")

        col4, col5, col6 = st.columns(3)

        position = col4.selectbox("Position", ["Long", "Short"], key=f"pos{i}")
        option_type = col5.selectbox("Type", ["Call", "Put"], key=f"type{i}")
        expiry = col6.date_input("Expiry", datetime.today(), key=f"exp{i}")

        if strike > 0:
            legs.append({
                "underlying": underlying,
                "strike": strike,
                "premium": premium,
                "position": position,
                "type": option_type,
                "expiry": expiry
            })

# -------------------------------
# Payoff Calculation
# -------------------------------
def calculate_payoff(legs):
    pct_range = np.linspace(-0.1, 0.1, 100)
    total_payoff = np.zeros_like(pct_range)
    greek_summary = []

    for leg in legs:
        S_base = nifty_price if leg["underlying"] == "NIFTY" else sensex_price
        S_range = S_base * (1 + pct_range)

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

        total_payoff += np.array(payoff)

        greeks = calculate_greeks(S_base, leg["strike"], T, r, sigma, leg["type"])
        greek_summary.append(greeks)

    return pct_range * 100, total_payoff, greek_summary


# -------------------------------
# Execute
# -------------------------------
if st.button("Calculate Payoff"):
    if len(legs) == 0:
        st.warning("Add at least one leg.")
    else:
        x, payoff, greeks = calculate_payoff(legs)

        df = pd.DataFrame({
            "Price Change (%)": x,
            "P&L": payoff
        })

        st.subheader("Payoff Chart")
        st.line_chart(df.set_index("Price Change (%)"))

        st.subheader("Greeks Summary")
        greek_df = pd.DataFrame(
            greeks,
            columns=["Delta", "Gamma", "Theta", "Vega", "Rho"]
        )
        st.dataframe(greek_df)
