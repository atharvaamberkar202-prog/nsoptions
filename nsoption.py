import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import yfinance as yf

st.set_page_config(layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
body {background-color:#090b0f; color:#e8edf3;}
.block-container {padding-top:1rem;}
.header-box {
    display:flex; justify-content:space-between; align-items:center;
    background:#0f1318; padding:12px 20px; border-radius:10px;
    border:1px solid #252d38; margin-bottom:15px;
}
.metric-box {
    background:#161b22; padding:8px 16px; border-radius:6px;
    border:1px solid #252d38; text-align:right;
}
.metric-label {font-size:10px; color:#8a97a8;}
.metric-value {font-size:16px; font-weight:600;}
.section {
    background:#0f1318; border:1px solid #252d38;
    border-radius:10px; padding:15px; margin-bottom:15px;
}
.title {font-weight:700; font-size:14px; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

# ------------------ BS MODEL ------------------
def bs_price(S, K, T, r, sigma, t):
    if T <= 0:
        return max(0, S-K) if t=="Call" else max(0, K-S)
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if t=="Call":
        return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def greeks(S,K,T,r,sigma,t):
    if T<=0: return [0]*5
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    delta = norm.cdf(d1) if t=="Call" else norm.cdf(d1)-1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)))/365
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    rho = (K*T*np.exp(-r*T)*norm.cdf(d2 if t=="Call" else -d2))/100
    return delta,gamma,theta,vega,rho

# ------------------ LIVE DATA ------------------
def fetch_prices():
    n=yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]
    s=yf.Ticker("^BSESN").history(period="1d")["Close"].iloc[-1]
    return n,s

if "nifty" not in st.session_state:
    st.session_state.nifty=22000.0
    st.session_state.sensex=73000.0

if st.button("Fetch Live Prices"):
    n,s=fetch_prices()
    st.session_state.nifty=n
    st.session_state.sensex=s

# ------------------ HEADER ------------------
today=datetime.now().strftime("%d %b %Y")
st.markdown(f"""
<div class="header-box">
    <div><b>OptionsCraft</b> | {today}</div>
    <div style="display:flex;gap:10px;">
        <div class="metric-box">
            <div class="metric-label">NIFTY</div>
            <div class="metric-value">{st.session_state.nifty:.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">SENSEX</div>
            <div class="metric-value">{st.session_state.sensex:.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ MAIN GRID ------------------
left, right = st.columns([2,1])

# ------------------ LEFT PANEL ------------------
with left:
    st.markdown('<div class="section"><div class="title">Strategy Legs</div>', unsafe_allow_html=True)

    legs=[]
    for i in range(6):
        cols=st.columns(6)
        underlying=cols[0].selectbox("Underlying",["NIFTY","SENSEX"],key=f"u{i}")
        strike=cols[1].number_input("Strike",value=22000.0,key=f"k{i}")
        premium=cols[2].number_input("Premium",value=100.0,key=f"p{i}")
        pos=cols[3].selectbox("Pos",["Long","Short"],key=f"pos{i}")
        typ=cols[4].selectbox("Type",["Call","Put"],key=f"type{i}")
        exp=cols[5].date_input("Expiry",datetime.today(),key=f"exp{i}")

        legs.append({
            "u":underlying,"k":strike,"p":premium,
            "pos":pos,"type":typ,"exp":exp
        })

    calc=st.button("Calculate Payoff")
    st.markdown("</div>", unsafe_allow_html=True)

    if calc:
        pct=np.linspace(-0.1,0.1,100)
        total=np.zeros_like(pct)
        greek_sum=np.zeros(5)

        for l in legs:
            S=st.session_state.nifty if l["u"]=="NIFTY" else st.session_state.sensex
            T=max((l["exp"]-datetime.today().date()).days/365,0.0001)
            sigma=0.2

            curve=[]
            for p in pct:
                price=bs_price(S*(1+p),l["k"],T,0.065,sigma,l["type"])
                pnl=price-l["p"]
                if l["pos"]=="Short": pnl=-pnl
                curve.append(pnl)
            total+=np.array(curve)

            g=np.array(greeks(S,l["k"],T,0.065,sigma,l["type"]))
            if l["pos"]=="Short": g=-g
            greek_sum+=g

        df=pd.DataFrame({"% Move":pct*100,"P&L":total})

        st.markdown('<div class="section"><div class="title">Payoff Chart</div>', unsafe_allow_html=True)
        st.line_chart(df.set_index("% Move"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section"><div class="title">Portfolio Greeks</div>', unsafe_allow_html=True)
        st.write(pd.DataFrame([greek_sum],columns=["Delta","Gamma","Theta","Vega","Rho"]))
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ RIGHT PANEL ------------------
with right:
    st.markdown('<div class="section"><div class="title">Controls</div>', unsafe_allow_html=True)
    r=st.slider("Risk Free Rate %",0.0,10.0,6.5)/100
    iv=st.slider("IV Shift %",-50,50,0)/100
    time_shift=st.slider("Days Forward",0,60,0)
    st.markdown("</div>", unsafe_allow_html=True)
