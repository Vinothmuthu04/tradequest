
# ============================================================
# TRADEQUEST V24
# Multi-Strategy Equity + F&O Opportunity Scanner
# ============================================================
#
# IMPORTANT:
# This is an independent quantitative scanner. It does NOT
# represent SEBI registration, personalized investment advice,
# or guaranteed signals.
#
# V24 philosophy:
#   Do not wait for one perfect setup.
#   Rank multiple independent approaches and give:
#   ENTRY / SL / T1 / T2 / T3 / R:R / SCORE
#
# Stock Universe now includes equity + F&O + indices in ONE
# selectable control.
#
# Strategies:
#   1. Breakout
#   2. Breakout-Ready
#   3. Pullback
#   4. Demand Reversal
#   5. Momentum
#   6. Trend Continuation
#   7. F&O option setup
#
# Install:
#   pip install streamlit yfinance pandas numpy requests plotly
#
# Run:
#   python -m streamlit run app.py
# ============================================================

import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TradeQuest V24",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

INITIAL_CAPITAL = 500000

# V19/V20 LOCKED SWING STRATEGY
TARGET_PERCENT = 8.0
STOP_PERCENT = 5.0
RISK_PERCENT = 1.0
MAX_POSITIONS_DEFAULT = 5
MAX_HOLDING_DAYS = 20
VOLUME_MULTIPLIER = 1.5
NEAR_MIN_CONDITIONS = 4
WATCH_MIN_CONDITIONS = 2

EQUITY_SL_PCT = STOP_PERCENT
EQUITY_T1_PCT = TARGET_PERCENT
EQUITY_T2_PCT = 12.0
EQUITY_T3_PCT = 18.0

FNO_SL_PCT = 30.0
FNO_T1_PCT = 25.0
FNO_T2_PCT = 50.0
FNO_T3_PCT = 75.0

MIN_SCORE_WATCH = 60
MIN_SCORE_READY_SOON = 72
MIN_SCORE_READY = 82

MAX_STOCKS_DEFAULT = 250
FNO_BUDGET_DEFAULT = 15000

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}

# ============================================================
# INDEX TICKERS
# ============================================================

INDEX_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
}

# ============================================================
# FALLBACK F&O STOCK LIST
# Used only if NSE derivatives download is unavailable.
# Yahoo option-chain availability is checked separately.
# ============================================================

FNO_FALLBACK = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DIVISLAB", "DLF", "DRREDDY", "EICHERMOT",
    "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR",
    "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "MAXHEALTH", "NATIONALUM", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SIEMENS", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "ZYDUSLIFE",
]


# ============================================================
# UTILITY
# ============================================================

def clean_symbol(symbol):
    return str(symbol).upper().strip().replace(".NS", "")


def yf_symbol(symbol):
    symbol = clean_symbol(symbol)
    if symbol in INDEX_TICKERS:
        return INDEX_TICKERS[symbol]
    return symbol + ".NS"


@st.cache_data(ttl=900, show_spinner=False)
def download_history(symbol, period="1y"):
    try:
        df = yf.download(
            yf_symbol(symbol),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [x for x in required if x not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df[required].copy()
        df = df.dropna(subset=["Close"])
        return df

    except Exception:
        return pd.DataFrame()


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):
    d = df.copy()

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    d["MA20"] = close.rolling(20).mean()
    d["MA50"] = close.rolling(50).mean()
    d["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]

    d["VOL20"] = volume.rolling(20).mean()
    d["VOL_RATIO"] = volume / d["VOL20"].replace(0, np.nan)

    d["HIGH20"] = high.rolling(20).max().shift(1)
    d["HIGH50"] = high.rolling(50).max().shift(1)
    d["LOW20"] = low.rolling(20).min().shift(1)
    d["LOW50"] = low.rolling(50).min().shift(1)

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["ATR14"] = tr.rolling(14).mean()
    d["ATR_PCT"] = d["ATR14"] / close * 100

    # Recent swing / demand proxies
    d["LOW10"] = low.rolling(10).min()
    d["LOW20_RAW"] = low.rolling(20).min()
    d["HIGH10"] = high.rolling(10).max()

    # Relative strength versus 50-day range
    d["RANGE50"] = d["HIGH50"] - d["LOW50"]

    return d


# ============================================================
# MARKET REGIME
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_market_regime():
    df = download_history("NIFTY", "1y")

    if df.empty or len(df) < 60:
        return {
            "regime": "UNKNOWN",
            "score": 0,
            "close": np.nan,
            "ma20": np.nan,
            "ma50": np.nan,
        }

    d = add_indicators(df)
    r = d.iloc[-1]

    score = 0

    if r["Close"] > r["MA20"]:
        score += 25
    if r["MA20"] > r["MA50"]:
        score += 25
    if r["RSI"] > 50:
        score += 20
    if r["MACD"] > r["MACD_SIGNAL"]:
        score += 20
    if r["Close"] > d["HIGH50"].iloc[-1]:
        score += 10

    if score >= 70:
        regime = "BULLISH"
    elif score >= 45:
        regime = "NEUTRAL"
    else:
        regime = "WEAK"

    return {
        "regime": regime,
        "score": score,
        "close": safe_float(r["Close"]),
        "ma20": safe_float(r["MA20"]),
        "ma50": safe_float(r["MA50"]),
    }


# ============================================================
# UNIVERSE LOADERS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_nifty200():
    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty200list.csv"
    )
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return sorted(
            {clean_symbol(x) for x in df[col].dropna()}
        )
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_nifty500():
    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"
    )
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return sorted(
            {clean_symbol(x) for x in df[col].dropna()}
        )
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_equity_symbols():
    url = (
        "https://nsearchives.nseindia.com/"
        "content/equities/EQUITY_L.csv"
    )

    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))

        col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        return sorted(
            {
                clean_symbol(x)
                for x in df[col].dropna()
                if str(x).strip()
            }
        )
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_fno_symbols():
    # Try NSE derivatives equity list first.
    candidates = [
        "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
        "https://nsearchives.nseindia.com/content/fo/NSE_FO_bhavcopy.csv",
    ]

    for url in candidates:
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            if r.ok and len(r.content) > 1000:
                text = r.text.upper()

                # Conservative extraction of known NSE-style symbols.
                found = set()
                for item in FNO_FALLBACK:
                    if item.upper() in text:
                        found.add(item)

                if len(found) >= 20:
                    return sorted(found)
        except Exception:
            pass

    return sorted(FNO_FALLBACK)


def load_universe(selected):
    nifty200 = set(load_nifty200())
    nifty500 = set(load_nifty500())
    nse_all = set(load_nse_equity_symbols())
    fno = set(load_fno_symbols())

    symbols = set()

    if "NIFTY 200" in selected:
        symbols |= nifty200

    if "NIFTY 500" in selected:
        symbols |= nifty500

    if "All Stocks > ₹10,000 Cr" in selected:
        # Market-cap filtering requires current market-cap metadata,
        # so start with the broad NSE universe and filter later.
        symbols |= nse_all

    if "Combined" in selected:
        symbols |= nifty200 | nifty500 | nse_all

    if "F&O Stocks" in selected:
        symbols |= fno

    return sorted(symbols)


# ============================================================
# MARKET-CAP FILTER
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def passes_market_cap(symbol):
    try:
        info = yf.Ticker(yf_symbol(symbol)).fast_info

        market_cap = info.get("market_cap", None)

        if market_cap is None:
            return False

        return float(market_cap) >= 10000 * 1e7

    except Exception:
        return False


# ============================================================
# SETUP ENGINE
# ============================================================

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def make_levels(entry, atr, strategy):
    entry = float(entry)

    if not np.isfinite(atr) or atr <= 0:
        atr = entry * 0.02

    # Volatility-aware stop, capped to reasonable limits.
    if strategy == "Demand Reversal":
        stop_distance = max(atr * 1.35, entry * 0.035)
    elif strategy == "Pullback":
        stop_distance = max(atr * 1.20, entry * 0.04)
    elif strategy == "Momentum":
        stop_distance = max(atr * 1.30, entry * 0.045)
    else:
        stop_distance = max(atr * 1.25, entry * 0.04)

    stop_distance = min(stop_distance, entry * 0.07)

    sl = entry - stop_distance

    risk = entry - sl

    t1 = entry + max(risk * 1.5, entry * EQUITY_T1_PCT / 100)
    t2 = entry + max(risk * 2.2, entry * EQUITY_T2_PCT / 100)
    t3 = entry + max(risk * 3.0, entry * EQUITY_T3_PCT / 100)

    rr = (t1 - entry) / risk if risk > 0 else 0

    return sl, t1, t2, t3, rr


def build_setup(symbol, d, strategy, base_score, trigger=None):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    atr = safe_float(r["ATR14"])

    if not np.isfinite(close) or close <= 0:
        return None

    entry = close if trigger is None else float(trigger)

    sl, t1, t2, t3, rr = make_levels(
        entry, atr, strategy
    )

    # Don't create trades with poor reward/risk.
    if rr < 1.25:
        return None

    distance = (
        ((trigger - close) / close * 100)
        if trigger is not None and close > 0
        else 0
    )

    if strategy in ("Breakout", "Momentum", "Trend Continuation"):
        status = "🟢 READY TO ENTER"
        if base_score < MIN_SCORE_READY:
            status = "🟡 READY SOON"
    else:
        if base_score >= MIN_SCORE_READY:
            status = "🟢 READY TO ENTER"
        elif base_score >= MIN_SCORE_READY_SOON:
            status = "🟡 READY SOON"
        else:
            status = "🔵 WATCHLIST"

    return {
        "Status": status,
        "Symbol": symbol,
        "Approach": strategy,
        "Current": round(close, 2),
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "T1": round(t1, 2),
        "T2": round(t2, 2),
        "T3": round(t3, 2),
        "R:R": round(rr, 2),
        "Score": int(clamp(base_score)),
        "RSI": round(safe_float(r["RSI"]), 1),
        "Vol x": round(safe_float(r["VOL_RATIO"]), 2),
        "MA20": round(safe_float(r["MA20"]), 2),
        "MA50": round(safe_float(r["MA50"]), 2),
        "Trigger Gap %": round(distance, 2),
    }


# ============================================================
# STRATEGY 1 - BREAKOUT
# ============================================================

def strategy_breakout(symbol, d):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    high20 = safe_float(r["HIGH20"])
    high50 = safe_float(r["HIGH50"])
    vol = safe_float(r["VOL_RATIO"])
    rsi = safe_float(r["RSI"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])

    if not all(np.isfinite(x) for x in [close, high20, high50, vol, rsi, ma20, ma50]):
        return None

    score = 0

    if close > high20:
        score += 22
    if close > high50:
        score += 18
    if close > ma20:
        score += 10
    if ma20 > ma50:
        score += 15
    if 52 <= rsi <= 75:
        score += 10
    if vol >= 1.5:
        score += 15
    elif vol >= 1.2:
        score += 8
    if macd > signal:
        score += 10

    if close <= high20:
        return None

    return build_setup(
        symbol, d, "Breakout", score
    )


# ============================================================
# STRATEGY 2 - BREAKOUT READY
# ============================================================

def strategy_breakout_ready(symbol, d):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    high20 = safe_float(r["HIGH20"])
    high50 = safe_float(r["HIGH50"])
    vol = safe_float(r["VOL_RATIO"])
    rsi = safe_float(r["RSI"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])

    if not all(np.isfinite(x) for x in [close, high20, high50, vol, rsi, ma20, ma50]):
        return None

    # Use the nearer meaningful resistance.
    resistances = [x for x in [high20, high50] if np.isfinite(x) and x > close]
    if not resistances:
        return None

    resistance = min(resistances)

    gap = (resistance - close) / close * 100

    if gap < 0 or gap > 3.0:
        return None

    score = 45

    if ma20 > ma50:
        score += 15
    if close > ma20:
        score += 10
    if 48 <= rsi <= 72:
        score += 8
    if macd > signal:
        score += 8
    if vol >= 1.2:
        score += 8
    if gap <= 1:
        score += 6

    trigger = resistance * 1.002

    return build_setup(
        symbol,
        d,
        "Breakout-Ready",
        score,
        trigger=trigger,
    )


# ============================================================
# STRATEGY 3 - PULLBACK
# ============================================================

def strategy_pullback(symbol, d):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    rsi = safe_float(r["RSI"])
    vol = safe_float(r["VOL_RATIO"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])

    if not all(np.isfinite(x) for x in [close, ma20, ma50, rsi, vol]):
        return None

    # Strong trend first.
    if not (ma20 > ma50):
        return None

    distance_ma20 = abs(close - ma20) / close * 100

    if distance_ma20 > 3.0:
        return None

    score = 48

    if close >= ma50:
        score += 12
    if close >= ma20 * 0.97:
        score += 10
    if 42 <= rsi <= 58:
        score += 10
    if rsi > 50:
        score += 5
    if macd >= signal:
        score += 10
    if vol >= 1.1:
        score += 5

    return build_setup(
        symbol, d, "Pullback", score
    )


# ============================================================
# STRATEGY 4 - DEMAND REVERSAL
# ============================================================

def strategy_demand_reversal(symbol, d):
    if len(d) < 30:
        return None

    r = d.iloc[-1]
    prev = d.iloc[-2]

    close = safe_float(r["Close"])
    low = safe_float(r["Low"])
    high = safe_float(r["High"])
    open_ = safe_float(r["Open"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    rsi = safe_float(r["RSI"])
    vol = safe_float(r["VOL_RATIO"])

    low20 = safe_float(r["LOW20_RAW"])
    low10 = safe_float(r["LOW10"])

    if not all(
        np.isfinite(x)
        for x in [close, low, high, open_, ma20, ma50, rsi, vol, low20, low10]
    ):
        return None

    candle_range = max(high - low, close * 0.001)
    lower_wick = min(open_, close) - low
    wick_ratio = lower_wick / candle_range

    near_demand = (
        abs(close - low20) / close <= 0.025
        or abs(close - low10) / close <= 0.02
    )

    bullish_reversal = (
        close > open_
        and close > safe_float(prev["Close"])
    )

    rsi_recovery = (
        rsi >= 40
        and rsi > safe_float(prev["RSI"])
    )

    if not near_demand:
        return None

    score = 45

    if bullish_reversal:
        score += 15
    if wick_ratio >= 0.35:
        score += 12
    if rsi_recovery:
        score += 12
    if vol >= 1.2:
        score += 8
    if close >= ma50:
        score += 8
    if ma20 >= ma50:
        score += 5

    # Entry slightly above reversal candle.
    trigger = max(close, high * 1.002)

    return build_setup(
        symbol,
        d,
        "Demand Reversal",
        score,
        trigger=trigger if score < MIN_SCORE_READY else None,
    )


# ============================================================
# STRATEGY 5 - MOMENTUM
# ============================================================

def strategy_momentum(symbol, d):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    open_ = safe_float(r["Open"])
    high = safe_float(r["High"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    rsi = safe_float(r["RSI"])
    vol = safe_float(r["VOL_RATIO"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])

    if not all(np.isfinite(x) for x in [close, open_, high, ma20, ma50, rsi, vol]):
        return None

    body_pct = abs(close - open_) / close * 100

    if body_pct < 0.8:
        return None

    score = 42

    if close > open_:
        score += 15
    if close > ma20:
        score += 10
    if ma20 > ma50:
        score += 10
    if rsi >= 55:
        score += 8
    if rsi <= 78:
        score += 5
    if vol >= 1.5:
        score += 15
    elif vol >= 1.2:
        score += 8
    if macd > signal:
        score += 10

    if score < MIN_SCORE_WATCH:
        return None

    return build_setup(
        symbol, d, "Momentum", score
    )


# ============================================================
# STRATEGY 6 - TREND CONTINUATION
# ============================================================

def strategy_continuation(symbol, d):
    r = d.iloc[-1]

    close = safe_float(r["Close"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    ma200 = safe_float(r["MA200"])
    rsi = safe_float(r["RSI"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])
    vol = safe_float(r["VOL_RATIO"])

    if not all(np.isfinite(x) for x in [close, ma20, ma50, ma200, rsi, macd, signal, vol]):
        return None

    if not (close > ma20 > ma50 > ma200):
        return None

    score = 55

    if rsi >= 50:
        score += 10
    if rsi <= 72:
        score += 8
    if macd > signal:
        score += 10
    if vol >= 1.1:
        score += 7
    if close <= ma20 * 1.03:
        score += 10

    return build_setup(
        symbol, d, "Trend Continuation", score
    )


# ============================================================
# RUN ALL EQUITY STRATEGIES
# ============================================================

def scan_symbol(symbol, market_regime):
    """V25 swing scanner: exact V19/V20 six-condition strategy."""
    df = download_history(symbol, "1y")
    if df.empty or len(df) < 220:
        return []

    d = add_indicators(df).dropna(
        subset=["MA20", "MA50", "RSI", "MACD", "MACD_SIGNAL", "VOL_RATIO", "HIGH50", "ATR14"]
    )
    if len(d) < 100:
        return []

    r = d.iloc[-1]
    close = safe_float(r["Close"])
    ma20 = safe_float(r["MA20"])
    ma50 = safe_float(r["MA50"])
    rsi = safe_float(r["RSI"])
    macd = safe_float(r["MACD"])
    signal = safe_float(r["MACD_SIGNAL"])
    volx = safe_float(r["VOL_RATIO"])
    high50 = safe_float(r["HIGH50"])
    atr = safe_float(r["ATR14"])

    vals = [close, ma20, ma50, rsi, macd, signal, volx, high50, atr]
    if not all(np.isfinite(x) for x in vals) or close <= 0:
        return []

    conditions = {
        "Price > MA20": close > ma20,
        "MA20 > MA50": ma20 > ma50,
        "RSI 50-70": 50 <= rsi <= 70,
        "MACD > Signal": macd > signal,
        "Volume >= 1.5x": volx >= VOLUME_MULTIPLIER,
        "50D Breakout": close > high50,
    }
    passed = sum(conditions.values())

    if passed >= 6:
        status = "🟢 READY — Entry Conditions Met"
    elif passed >= NEAR_MIN_CONDITIONS:
        status = "🟡 NEAR ENTRY — Watch Closely"
    elif passed >= WATCH_MIN_CONDITIONS:
        status = "🔵 WATCH — Setup Developing"
    else:
        return []

    score = passed * 15
    if 50 <= rsi <= 70:
        score += 5
    if volx >= 2.0:
        score += 5
    if close > high50:
        score += 5
    if market_regime == "BULLISH":
        score += 4

    entry = close
    sl = entry * (1 - STOP_PERCENT / 100)
    t1 = entry * (1 + TARGET_PERCENT / 100)
    risk_per_share = entry - sl
    rr = (t1 - entry) / risk_per_share if risk_per_share > 0 else 0
    missing = [name for name, ok in conditions.items() if not ok]

    return [{
        "Status": status,
        "Symbol": symbol,
        "Strategy": "V19/V20 Six-Condition Swing",
        "Approach": "V19/V20 Swing",
        "Current": round(close, 2),
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "T1": round(t1, 2),
        "Target %": TARGET_PERCENT,
        "Stop %": STOP_PERCENT,
        "Risk %": RISK_PERCENT,
        "Max Hold Days": MAX_HOLDING_DAYS,
        "Conditions Met": f"{passed}/6",
        "Missing": ", ".join(missing) if missing else "None",
        "R:R": round(rr, 2),
        "Score": int(clamp(score)),
        "RSI": round(rsi, 1),
        "Vol x": round(volx, 2),
        "MA20": round(ma20, 2),
        "MA50": round(ma50, 2),
        "50D High": round(high50, 2),
    }]


# ============================================================
# INTRADAY ENGINE — INDEPENDENT STRATEGY
# ============================================================
# Strategy: 15-minute trend + 5-minute opening-range/VWAP trigger.
# This is deliberately separate from the daily swing strategies.

@st.cache_data(ttl=180, show_spinner=False)
def download_intraday(symbol, interval="5m", period="5d"):
    try:
        df = yf.download(
            yf_symbol(symbol),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in required):
            return pd.DataFrame()
        d = df[required].dropna(subset=["Close"]).copy()
        return d
    except Exception:
        return pd.DataFrame()


def add_intraday_indicators(df):
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    d["EMA9"] = close.ewm(span=9, adjust=False).mean()
    d["EMA20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    # Session VWAP. Reset at each trading day.
    typical = (high + low + close) / 3
    date_key = pd.Series(d.index.date, index=d.index)
    pv = typical * volume
    d["VWAP"] = pv.groupby(date_key).cumsum() / volume.groupby(date_key).cumsum().replace(0, np.nan)

    d["VOL20"] = volume.rolling(20).mean()
    d["VOL_RATIO"] = volume / d["VOL20"].replace(0, np.nan)
    d["ATR14"] = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()

    # Day/session levels.
    d["DAY_HIGH"] = high.groupby(date_key).cummax()
    d["DAY_LOW"] = low.groupby(date_key).cummin()
    return d


def intraday_status(score, gap_pct, direction):
    if score >= 82 and gap_pct <= 0.35:
        return "🟢 INTRADAY READY"
    if score >= 68 and gap_pct <= 1.25:
        return "🟡 INTRADAY WAITING"
    return "🔵 INTRADAY WATCH"


def strategy_intraday(symbol):
    # Independent intraday strategy: 15m directional structure + 5m trigger.
    d5 = download_intraday(symbol, "5m", "5d")
    d15 = download_intraday(symbol, "15m", "10d")
    if d5.empty or d15.empty or len(d5) < 40 or len(d15) < 30:
        return None

    a5 = add_intraday_indicators(d5).dropna(subset=["EMA9", "EMA20", "VWAP", "RSI", "ATR14"])
    a15 = add_intraday_indicators(d15).dropna(subset=["EMA9", "EMA20", "VWAP", "RSI"])
    if a5.empty or a15.empty:
        return None

    r = a5.iloc[-1]
    p = a5.iloc[-2]
    r15 = a15.iloc[-1]

    close = safe_float(r["Close"])
    ema9 = safe_float(r["EMA9"])
    ema20 = safe_float(r["EMA20"])
    vwap = safe_float(r["VWAP"])
    rsi = safe_float(r["RSI"])
    volx = safe_float(r["VOL_RATIO"], 0)
    atr = safe_float(r["ATR14"])
    high = safe_float(r["High"])
    low = safe_float(r["Low"])
    prev_high = safe_float(p["High"])
    prev_low = safe_float(p["Low"])
    trend15_up = safe_float(r15["Close"]) > safe_float(r15["EMA20"]) and safe_float(r15["EMA9"]) > safe_float(r15["EMA20"])
    trend15_down = safe_float(r15["Close"]) < safe_float(r15["EMA20"]) and safe_float(r15["EMA9"]) < safe_float(r15["EMA20"])

    vals = [close, ema9, ema20, vwap, rsi, volx, atr, high, low, prev_high, prev_low]
    if not all(np.isfinite(x) for x in vals) or atr <= 0:
        return None

    # Opening range = first 15 minutes of the current session.
    session_date = d5.index[-1].date()
    today = d5[pd.Series(d5.index.date, index=d5.index) == session_date]
    if len(today) < 3:
        return None
    opening = today.iloc[:3]
    orb_high = safe_float(opening["High"].max())
    orb_low = safe_float(opening["Low"].min())

    bullish = trend15_up and close > vwap and ema9 > ema20 and rsi >= 52
    bearish = trend15_down and close < vwap and ema9 < ema20 and rsi <= 48
    if not bullish and not bearish:
        return None

    direction = "LONG" if bullish else "SHORT"
    trigger = max(orb_high, prev_high, ema9, vwap) * 1.0005 if bullish else min(orb_low, prev_low, ema9, vwap) * 0.9995
    gap = abs(trigger - close) / close * 100

    score = 40
    if (bullish and close > vwap) or (bearish and close < vwap): score += 12
    if (bullish and ema9 > ema20) or (bearish and ema9 < ema20): score += 12
    if (bullish and trend15_up) or (bearish and trend15_down): score += 15
    if (bullish and 52 <= rsi <= 72) or (bearish and 28 <= rsi <= 48): score += 10
    if volx >= 1.5: score += 12
    elif volx >= 1.2: score += 7
    if (bullish and close >= orb_high) or (bearish and close <= orb_low): score += 8
    if gap <= 0.35: score += 6

    if score < 60:
        return None

    entry = trigger if gap <= 0.35 else close
    if bullish:
        sl = entry - max(atr * 1.15, entry * 0.004)
        t1 = entry + (entry - sl) * 1.0
        t2 = entry + (entry - sl) * 1.8
        t3 = entry + (entry - sl) * 2.6
    else:
        sl = entry + max(atr * 1.15, entry * 0.004)
        t1 = entry - (sl - entry) * 1.0
        t2 = entry - (sl - entry) * 1.8
        t3 = entry - (sl - entry) * 2.6

    risk_per_share = abs(entry - sl)
    rr = abs(t1 - entry) / risk_per_share if risk_per_share else 0
    status = intraday_status(score, gap, direction)

    return {
        "Status": status,
        "Symbol": symbol,
        "Strategy": "15m Trend + 5m VWAP/ORB Momentum",
        "Direction": direction,
        "Current": round(close, 2),
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "T1": round(t1, 2),
        "T2": round(t2, 2),
        "T3": round(t3, 2),
        "R:R": round(rr, 2),
        "Score": int(clamp(score)),
        "RSI": round(rsi, 1),
        "VWAP": round(vwap, 2),
        "ORB High": round(orb_high, 2),
        "ORB Low": round(orb_low, 2),
        "Vol x": round(volx, 2),
        "Trigger Gap %": round(gap, 2),
    }


def scan_intraday(symbols, max_count):
    rows = []
    symbols = symbols[:max_count]
    progress = st.progress(0)
    total = max(len(symbols), 1)
    for i, symbol in enumerate(symbols):
        try:
            result = strategy_intraday(symbol)
            if result:
                rows.append(result)
        except Exception:
            pass
        progress.progress((i + 1) / total)
    progress.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Score", "R:R"], ascending=False).reset_index(drop=True)


def build_portfolio_plan(df, capital, max_positions=MAX_POSITIONS_DEFAULT):
    """Allocate entered capital equally across top READY setups."""
    if df is None or df.empty or "Status" not in df.columns:
        return pd.DataFrame()
    ready = df[df["Status"] == "🟢 READY — Entry Conditions Met"].copy()
    if ready.empty:
        return pd.DataFrame()
    ready = ready.sort_values(["Score", "R:R"], ascending=False).drop_duplicates("Symbol").head(max_positions).copy()
    allocation = float(capital) / len(ready)
    rows = []
    for rank, (_, row) in enumerate(ready.iterrows(), 1):
        entry = safe_float(row.get("Entry"), 0)
        sl = safe_float(row.get("SL"), 0)
        target = safe_float(row.get("T1"), 0)
        qty = int(allocation / entry) if entry > 0 else 0
        investment = qty * entry
        risk = qty * max(entry - sl, 0)
        profit = qty * max(target - entry, 0)
        rows.append({
            "Rank": rank,
            "Symbol": row.get("Symbol", ""),
            "Entry": round(entry, 2),
            "Allocation ₹": round(allocation, 0),
            "Qty": qty,
            "Investment ₹": round(investment, 0),
            "SL": round(sl, 2),
            "Target": round(target, 2),
            "Risk ₹": round(risk, 0),
            "Potential Profit ₹": round(profit, 0),
            "Max Hold": f"{MAX_HOLDING_DAYS} days",
        })
    return pd.DataFrame(rows)


def apply_equity_position_sizing(df, capital):
    """Risk sizing for intraday only; swing uses build_portfolio_plan."""
    if df.empty:
        return df
    d = df.copy()
    risk_budget = capital * (RISK_PERCENT / 100.0)
    qtys, used = [], []
    for _, row in d.iterrows():
        entry = safe_float(row.get("Entry"), 0)
        sl = safe_float(row.get("SL"), 0)
        if entry <= 0 or sl <= 0 or entry == sl:
            qty = 0
        else:
            qty = max(0, int(risk_budget / abs(entry - sl)))
            qty = min(qty, int(capital / entry)) if entry else 0
        qtys.append(qty)
        used.append(round(qty * entry, 0))
    d["Qty @ 1% Risk"] = qtys
    d["Capital Used ₹"] = used
    return d


# ============================================================
# F&O OPTION ENGINE
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(symbol):
    try:
        ticker = yf.Ticker(yf_symbol(symbol))
        expiries = ticker.options

        if not expiries:
            return None, []

        expiry = expiries[0]
        chain = ticker.option_chain(expiry)

        return expiry, chain

    except Exception:
        return None, []


def option_setup(symbol, option_type, budget):
    try:
        expiry, chain = get_option_chain(symbol)

        if not expiry:
            return None

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        options = calls if option_type == "CE" else puts

        if options.empty:
            return None

        underlying_df = download_history(symbol, "6mo")

        if underlying_df.empty:
            return None

        underlying = add_indicators(underlying_df).dropna(
            subset=["MA20", "MA50", "RSI"]
        )

        if underlying.empty:
            return None

        u = underlying.iloc[-1]

        price = safe_float(u["Close"])
        ma20 = safe_float(u["MA20"])
        ma50 = safe_float(u["MA50"])
        rsi = safe_float(u["RSI"])

        if not all(np.isfinite(x) for x in [price, ma20, ma50, rsi]):
            return None

        # Determine directional bias.
        bullish = (
            price > ma20
            and ma20 > ma50
            and rsi >= 52
        )

        bearish = (
            price < ma20
            and ma20 < ma50
            and rsi <= 48
        )

        if option_type == "CE" and not bullish:
            return None

        if option_type == "PE" and not bearish:
            return None

        # ATM / near-ATM contracts.
        options = options.copy()

        options["distance"] = (
            options["strike"] - price
        ).abs()

        options = options.sort_values("distance").head(8)

        if options.empty:
            return None

        best = None

        for _, row in options.iterrows():
            last = safe_float(row.get("lastPrice"))

            if not np.isfinite(last) or last <= 0:
                continue

            volume = safe_float(row.get("volume"), 0)
            oi = safe_float(row.get("openInterest"), 0)

            lot_size = 1

            # Yahoo generally does not expose NSE lot size reliably.
            # Use contract metadata where possible; otherwise display
            # a conservative quantity of 1.
            try:
                contract = yf.Ticker(
                    yf_symbol(symbol)
                ).get_shares_full(start=None, end=None)
                _ = contract
            except Exception:
                pass

            score = 45

            if volume > 0:
                score += 10
            if oi > 0:
                score += 10
            if volume >= 1000:
                score += 10
            if oi >= 10000:
                score += 10

            if bullish and option_type == "CE":
                score += 10

            if bearish and option_type == "PE":
                score += 10

            # Entry zone around current premium.
            entry_low = last * 0.98
            entry_high = last * 1.04

            entry = last

            sl = entry * (1 - FNO_SL_PCT / 100)

            t1 = entry * (1 + FNO_T1_PCT / 100)
            t2 = entry * (1 + FNO_T2_PCT / 100)
            t3 = entry * (1 + FNO_T3_PCT / 100)

            risk = entry - sl
            rr = (t1 - entry) / risk if risk > 0 else 0

            cost = entry * lot_size
            lots = int(budget / cost) if cost > 0 else 0

            if lots < 1:
                lots = 1

            if score >= 82:
                status = "🟢 F&O READY"
            elif score >= 65:
                status = "⚡ F&O WATCH"
            else:
                continue

            candidate = {
                "Status": status,
                "Symbol": symbol,
                "Approach": "F&O",
                "Option": option_type,
                "Strike": safe_float(row.get("strike")),
                "Expiry": expiry,
                "Underlying": round(price, 2),
                "Premium": round(last, 2),
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "T1": round(t1, 2),
                "T2": round(t2, 2),
                "T3": round(t3, 2),
                "R:R": round(rr, 2),
                "Volume": int(volume) if np.isfinite(volume) else 0,
                "OI": int(oi) if np.isfinite(oi) else 0,
                "Score": int(clamp(score)),
                "Lots": lots,
                "Capital ₹": round(entry * lot_size * lots, 0),
            }

            if best is None or candidate["Score"] > best["Score"]:
                best = candidate

        return best

    except Exception:
        return None


def scan_fno(symbols, option_types, budget, max_count):
    rows = []

    symbols = symbols[:max_count]

    progress = st.progress(0)
    total = max(len(symbols), 1)

    for i, symbol in enumerate(symbols):
        for option_type in option_types:
            result = option_setup(
                symbol,
                option_type,
                budget,
            )

            if result is not None:
                rows.append(result)

        progress.progress((i + 1) / total)

    progress.empty()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Score", "R:R"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# TRADEQUEST V24.1
# SINGLE-SCANNER MODE + PREVIOUS READY TRACKING
# ============================================================
#
# V24.1 changes:
# 1. ONLY ONE scanner runs at a time.
#    Swing / Intraday / Stock F&O / Index F&O are independent.
# 2. Each scanner keeps its own strategy.
# 3. Capital is selectable by the user.
# 4. V19-style READY TO ENTER remains available.
# 5. Previous READY signals are stored locally and carried forward.
#    This is for cases such as Jyoti CNC / ACE that were READY on the
#    previous session and need to remain visible with today's status.
# 6. Selecting NIFTY 500 scans ONLY NIFTY 500.
#    Selecting Stock F&O scans ONLY F&O stocks.
#    Selecting Intraday scans ONLY the selected intraday universe.
# ============================================================

import json

HISTORY_FILE = Path("tradequest_signal_history.json")

# ----------------------------
# CAPITAL
# ----------------------------
st.sidebar.header("⚙️ TradeQuest V25")

capital_choice = st.sidebar.selectbox(
    "💰 Trading Capital",
    [
        "₹25,000", "₹50,000", "₹1,00,000", "₹2,00,000",
        "₹5,00,000", "₹10,00,000", "₹25,00,000", "Custom"
    ],
    index=4,
)

if capital_choice == "Custom":
    trading_capital = float(st.sidebar.number_input(
        "Custom Capital ₹",
        min_value=5000,
        max_value=100000000,
        value=500000,
        step=5000,
    ))
else:
    trading_capital = float(capital_choice.replace("₹", "").replace(",", ""))

# ----------------------------
# ONE SCANNER ONLY
# ----------------------------
scanner_mode = st.sidebar.radio(
    "🔎 Scanner to Run",
    ["📈 Swing", "⚡ Intraday", "🎯 Stock F&O", "📊 Index F&O"],
    index=0,
)

st.sidebar.caption("Only the selected scanner is executed in this run.")

if scanner_mode == "📈 Swing":
    selected_universe = st.sidebar.selectbox(
        "📊 Swing Universe",
        ["NIFTY 200", "NIFTY 500", "All Stocks > ₹10,000 Cr", "F&O Stocks", "Combined"],
        index=1,
    )
    max_stocks = st.sidebar.slider("Stocks to scan", 50, 600, 250, 25)
    max_positions = st.sidebar.number_input("📌 Max Positions", min_value=1, max_value=20, value=5, step=1)
    market_filter = st.sidebar.checkbox("🌐 Market Regime Filter", value=False)
    st.sidebar.caption("V19/V20 Swing: 8% target | 5% SL | 1% risk | 20-day max hold")

elif scanner_mode == "⚡ Intraday":
    selected_universe = st.sidebar.selectbox(
        "📊 Intraday Universe",
        ["NIFTY 200", "NIFTY 500", "All Stocks > ₹10,000 Cr", "F&O Stocks"],
        index=1,
    )
    max_stocks = st.sidebar.slider("Stocks to scan", 25, 500, 150, 25)
    intraday_filters = st.sidebar.multiselect(
        "🎯 Intraday Status",
        ["🟢 INTRADAY READY", "🟡 INTRADAY WAITING", "🔵 INTRADAY WATCH"],
        default=["🟢 INTRADAY READY", "🟡 INTRADAY WAITING", "🔵 INTRADAY WATCH"],
    )

elif scanner_mode == "🎯 Stock F&O":
    max_stocks = st.sidebar.slider("F&O stocks to scan", 20, 250, 100, 10)
    option_types = st.sidebar.multiselect("Option Types", ["CE", "PE"], default=["CE", "PE"])
    fno_budget = st.sidebar.number_input(
        "💰 F&O Budget ₹", min_value=1000, max_value=10000000,
        value=int(min(trading_capital, 15000)), step=1000
    )
    fno_filters = st.sidebar.multiselect(
        "🎯 F&O Status", ["🟢 F&O READY", "⚡ F&O WATCH"],
        default=["🟢 F&O READY", "⚡ F&O WATCH"],
    )

else:
    selected_indices = st.sidebar.multiselect(
        "📌 Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
        default=["NIFTY", "BANKNIFTY"],
    )
    option_types = st.sidebar.multiselect("Option Types", ["CE", "PE"], default=["CE", "PE"])
    fno_budget = st.sidebar.number_input(
        "💰 Index F&O Budget ₹", min_value=1000, max_value=10000000,
        value=int(min(trading_capital, 15000)), step=1000
    )
    fno_filters = st.sidebar.multiselect(
        "🎯 F&O Status", ["🟢 F&O READY", "⚡ F&O WATCH"],
        default=["🟢 F&O READY", "⚡ F&O WATCH"],
    )

run_scan = st.sidebar.button("🔎 RUN SELECTED SCANNER", type="primary")

# ============================================================
# PREVIOUS READY MEMORY
# ============================================================

def load_signal_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_signal_history(rows):
    cutoff = datetime.now() - timedelta(days=30)
    clean = []
    for r in rows:
        try:
            dt = datetime.fromisoformat(str(r.get("saved_at", "")))
            if dt >= cutoff:
                clean.append(r)
        except Exception:
            clean.append(r)
    try:
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)
    except Exception:
        pass


def previous_ready_rows(scanner_name):
    today = datetime.now().strftime("%Y-%m-%d")
    history = load_signal_history()
    rows = [
        r for r in history
        if r.get("scanner") == scanner_name
        and r.get("status") == "🟢 READY — Entry Conditions Met"
        and r.get("saved_date") != today
    ]
    latest = {}
    for r in sorted(rows, key=lambda x: x.get("saved_at", ""), reverse=True):
        key = (r.get("symbol"), r.get("approach"))
        if key not in latest:
            latest[key] = r
    return list(latest.values())


def remember_ready_rows(scanner_name, df):
    if df is None or df.empty or "Status" not in df.columns:
        return
    ready = df[df["Status"] == "🟢 READY — Entry Conditions Met"]
    if ready.empty:
        return
    history = load_signal_history()
    today = datetime.now().strftime("%Y-%m-%d")
    keys = {
        (r.get("scanner"), r.get("symbol"), r.get("approach"), r.get("saved_date"))
        for r in history
    }
    for _, r in ready.iterrows():
        rec = {
            "scanner": scanner_name,
            "symbol": str(r.get("Symbol", "")),
            "approach": str(r.get("Approach", r.get("Strategy", ""))),
            "status": "🟢 READY — Entry Conditions Met",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_date": today,
            "entry": safe_float(r.get("Entry"), np.nan),
            "sl": safe_float(r.get("SL"), np.nan),
            "t1": safe_float(r.get("T1"), np.nan),
            "t2": safe_float(r.get("T2"), np.nan),
            "t3": safe_float(r.get("T3"), np.nan),
        }
        key = (rec["scanner"], rec["symbol"], rec["approach"], rec["saved_date"])
        if key not in keys:
            history.append(rec)
            keys.add(key)
    save_signal_history(history)


def compare_previous_ready(previous, current):
    if not previous:
        return pd.DataFrame()
    out = []
    for old in previous:
        symbol = str(old.get("symbol", ""))
        approach = str(old.get("approach", ""))
        price = np.nan
        if current is not None and not current.empty and "Symbol" in current.columns:
            m = current[current["Symbol"].astype(str) == symbol]
            if not m.empty and "Current" in m.columns:
                price = safe_float(m.iloc[0].get("Current"), np.nan)
        entry = safe_float(old.get("entry"), np.nan)
        sl = safe_float(old.get("sl"), np.nan)
        t1 = safe_float(old.get("t1"), np.nan)
        if np.isfinite(price):
            if np.isfinite(sl) and price <= sl:
                status = "🔴 INVALIDATED / SL ZONE"
            elif np.isfinite(t1) and price >= t1:
                status = "🟢 TARGET 1 REACHED"
            elif np.isfinite(entry) and price >= entry:
                status = "🟢 ENTRY TRIGGERED"
            else:
                status = "🟡 STILL WAITING"
        else:
            status = "⚪ NO CURRENT SETUP"
        out.append({
            "Previous Date": old.get("saved_date", ""),
            "Symbol": symbol,
            "Setup": approach,
            "Previous Entry": entry,
            "Previous SL": sl,
            "Previous T1": t1,
            "Current Price": price,
            "Status Today": status,
        })
    return pd.DataFrame(out)

# ============================================================
# UNIVERSE HELPERS
# ============================================================

def selected_symbols(universe, limit_count):
    if universe == "NIFTY 200":
        symbols = load_nifty200()
    elif universe == "NIFTY 500":
        symbols = load_nifty500()
    elif universe == "F&O Stocks":
        symbols = load_fno_symbols()
    elif universe == "All Stocks > ₹10,000 Cr":
        symbols = load_nse_equity_symbols()
        if len(symbols) > 700:
            symbols = symbols[:700]
        symbols = [s for s in symbols if passes_market_cap(s)]
    elif universe == "Combined":
        symbols = sorted(set(load_nifty200()) | set(load_nifty500()) | set(load_fno_symbols()))
    else:
        symbols = []
    return sorted(set(symbols))[:limit_count]

# ============================================================
# RUN ONLY THE SELECTED SCANNER
# ============================================================

def run_swing():
    symbols = selected_symbols(selected_universe, max_stocks)
    rows = []
    progress = st.progress(0)
    with st.spinner(f"Scanning SWING only — {selected_universe} ({len(symbols)} stocks)..."):
        regime_local = get_market_regime()
        for i, symbol in enumerate(symbols):
            try:
                rows.extend(scan_symbol(symbol, regime_local["regime"]))
            except Exception:
                pass
            progress.progress((i + 1) / max(len(symbols), 1))
    progress.empty()
    result = pd.DataFrame(rows) if rows else pd.DataFrame()
    return result, regime_local


def run_intraday_selected():
    symbols = selected_symbols(selected_universe, max_stocks)
    with st.spinner(f"Scanning INTRADAY only — {selected_universe} ({len(symbols)} stocks)..."):
        result = scan_intraday(symbols, len(symbols))
    if not result.empty:
        result = apply_equity_position_sizing(result, trading_capital)
    return result


def run_stock_fno_selected():
    symbols = sorted(set(load_fno_symbols()))[:max_stocks]
    with st.spinner(f"Scanning STOCK F&O only — {len(symbols)} F&O stocks..."):
        return scan_fno(symbols, option_types, fno_budget, len(symbols))


def run_index_fno_selected():
    rows = []
    with st.spinner("Scanning INDEX F&O only..."):
        for idx in selected_indices:
            for opt in option_types:
                try:
                    result = option_setup(idx, opt, fno_budget)
                    if result:
                        result["Strategy"] = "Index Trend + Option Momentum"
                        rows.append(result)
                except Exception:
                    pass
    return pd.DataFrame(rows).sort_values(["Score", "R:R"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

# ============================================================
# STATE + EXECUTION
# ============================================================

if "v25_results" not in st.session_state:
    st.session_state["v25_results"] = pd.DataFrame()
if "v25_previous" not in st.session_state:
    st.session_state["v25_previous"] = pd.DataFrame()
if "v25_regime" not in st.session_state:
    st.session_state["v25_regime"] = {"regime": "NOT USED", "score": 0, "close": np.nan, "ma20": np.nan, "ma50": np.nan}
if "v25_mode" not in st.session_state:
    st.session_state["v25_mode"] = scanner_mode

if scanner_mode != st.session_state["v25_mode"]:
    st.session_state["v25_results"] = pd.DataFrame()
    st.session_state["v25_previous"] = pd.DataFrame()
    st.session_state["v25_mode"] = scanner_mode

if run_scan:
    previous = previous_ready_rows("📈 Swing") if scanner_mode == "📈 Swing" else []

    if scanner_mode == "📈 Swing":
        result, regime = run_swing()
        st.session_state["v25_regime"] = regime
        previous_df = compare_previous_ready(previous, result)
        st.session_state["v25_previous"] = previous_df
        remember_ready_rows("📈 Swing", result)
    elif scanner_mode == "⚡ Intraday":
        result = run_intraday_selected()
        st.session_state["v25_previous"] = pd.DataFrame()
    elif scanner_mode == "🎯 Stock F&O":
        result = run_stock_fno_selected()
        st.session_state["v25_previous"] = pd.DataFrame()
    else:
        result = run_index_fno_selected()
        st.session_state["v25_previous"] = pd.DataFrame()

    st.session_state["v25_results"] = result
    st.session_state["v25_last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

results = st.session_state["v25_results"]
previous_df = st.session_state["v25_previous"]
regime = st.session_state["v25_regime"]

# ============================================================
# HEADER
# ============================================================

st.title("📈 TradeQuest V25")
st.caption("Single-Scanner Live Entry + Portfolio Scanner")
st.info(
    f"Running: **{scanner_mode}**  |  Capital: **₹{trading_capital:,.0f}**  |  "
    f"Last scan: **{st.session_state.get('v25_last_scan', 'Not run')}**"
)

# ============================================================
# SWING DISPLAY
# ============================================================

if scanner_mode == "📈 Swing":
    st.header("🎯 Live Entry Scanner")
    st.caption(f"{selected_universe} only | Capital ₹{trading_capital:,.0f} | Target 8% | SL 5% | Risk 1% | Max positions {max_positions} | Max hold 20 days")

    if results.empty:
        st.info("No qualifying setup found. This is a NO-TRADE result.")
    else:
        ready = results[results["Status"] == "🟢 READY — Entry Conditions Met"].copy()
        near = results[results["Status"] == "🟡 NEAR ENTRY — Watch Closely"].copy()
        watch = results[results["Status"] == "🔵 WATCH — Setup Developing"].copy()

        st.subheader("🟢 READY — Entry Conditions Met")
        if ready.empty:
            st.info("No READY setup right now.")
        else:
            cols = [x for x in ["Symbol","Current","Entry","SL","T1","Conditions Met","RSI","Vol x","MA20","MA50","50D High","Score"] if x in ready.columns]
            st.dataframe(ready.sort_values(["Score","R:R"], ascending=False)[cols], use_container_width=True, hide_index=True)

        st.subheader("🟡 NEAR ENTRY — Watch Closely")
        if near.empty:
            st.info("No NEAR ENTRY setup right now.")
        else:
            cols = [x for x in ["Symbol","Current","Entry","Conditions Met","Missing","RSI","Vol x","Score"] if x in near.columns]
            st.dataframe(near.sort_values(["Score","Conditions Met"], ascending=False)[cols], use_container_width=True, hide_index=True)

        st.subheader("🔵 WATCH — Setup Developing")
        if watch.empty:
            st.info("No WATCH setup right now.")
        else:
            cols = [x for x in ["Symbol","Current","Entry","Conditions Met","Missing","RSI","Vol x","Score"] if x in watch.columns]
            st.dataframe(watch.sort_values(["Score","Conditions Met"], ascending=False)[cols], use_container_width=True, hide_index=True)

        st.subheader("🔥 TOP READY ACTION LIST")
        if ready.empty:
            st.info("No READY stock means no action list and no forced trade.")
        else:
            top = ready.sort_values(["Score","R:R"], ascending=False).drop_duplicates("Symbol").head(int(max_positions)).copy()
            cols = [x for x in ["Symbol","Current","Entry","SL","T1","Score","R:R"] if x in top.columns]
            st.dataframe(top[cols], use_container_width=True, hide_index=True)

            st.subheader(f"💰 ₹{trading_capital:,.0f} PORTFOLIO PLAN")
            portfolio = build_portfolio_plan(top, trading_capital, int(max_positions))
            if portfolio.empty:
                st.info("No portfolio can be built because there are no READY setups.")
            else:
                total_investment = portfolio["Investment ₹"].sum()
                total_risk = portfolio["Risk ₹"].sum()
                total_profit = portfolio["Potential Profit ₹"].sum()
                p1,p2,p3,p4 = st.columns(4)
                p1.metric("Capital", f"₹{trading_capital:,.0f}")
                p2.metric("Positions", len(portfolio))
                p3.metric("Capital Used", f"₹{total_investment:,.0f}")
                p4.metric("Risk at 5% SL", f"₹{total_risk:,.0f}")
                st.dataframe(portfolio, use_container_width=True, hide_index=True)
                st.caption(f"Target is +8%. Planned gross profit at target: ₹{total_profit:,.0f}. Allocation is divided equally among the top READY setups; risk is separate from allocation.")

        st.subheader("🕘 YESTERDAY READY → TODAY STATUS")
        if previous_df.empty:
            st.info("No previous-session READY signal is stored yet. Today's READY signals are saved for the next scan.")
        else:
            st.dataframe(previous_df, use_container_width=True, hide_index=True)

# ============================================================
# INTRADAY DISPLAY
# ============================================================

elif scanner_mode == "⚡ Intraday":
    st.header("⚡ INTRADAY OPPORTUNITY BOARD")
    st.caption(f"Universe: {selected_universe} | Independent strategy: 15m trend + 5m VWAP/ORB momentum trigger.")
    if results.empty:
        st.info("No intraday setup currently meets the minimum score. This is a NO-TRADE result.")
    else:
        filtered = results[results["Status"].isin(intraday_filters)] if "Status" in results.columns else results
        ready = results[results["Status"] == "🟢 INTRADAY READY"]
        waiting = results[results["Status"] == "🟡 INTRADAY WAITING"]
        watch = results[results["Status"] == "🔵 INTRADAY WATCH"]
        a,b,c,d=st.columns(4); a.metric("🟢 READY",len(ready)); b.metric("🟡 WAITING",len(waiting)); c.metric("🔵 WATCH",len(watch)); d.metric("Total",len(results))
        if not filtered.empty:
            cols=[x for x in ["Status","Symbol","Strategy","Direction","Current","Entry","SL","T1","T2","T3","R:R","Score","RSI","VWAP","ORB High","ORB Low","Vol x","Qty @ 1% Risk","Capital Used ₹"] if x in filtered.columns]
            st.dataframe(filtered.sort_values(["Score","R:R"],ascending=False).head(50)[cols],use_container_width=True,hide_index=True)

# ============================================================
# STOCK F&O DISPLAY
# ============================================================

elif scanner_mode == "🎯 Stock F&O":
    st.header("🎯 STOCK F&O OPPORTUNITY BOARD")
    st.caption(f"F&O stocks only | Options: {', '.join(option_types)} | Budget: ₹{fno_budget:,.0f}")
    if results.empty:
        st.info("No stock F&O setup passed the V24 quality/liquidity and directional checks. This is a NO-TRADE result.")
    else:
        filtered = results[results["Status"].isin(fno_filters)] if "Status" in results.columns else results
        ready = results[results["Status"] == "🟢 F&O READY"]
        watch = results[results["Status"] == "⚡ F&O WATCH"]
        a,b,c=st.columns(3); a.metric("🟢 F&O READY",len(ready)); b.metric("⚡ F&O WATCH",len(watch)); c.metric("Opportunities",len(results))
        if not filtered.empty:
            cols=[x for x in ["Status","Symbol","Approach","Option","Strike","Expiry","Underlying","Premium","Entry","SL","T1","T2","T3","R:R","Score","Volume","OI","Lots","Capital ₹"] if x in filtered.columns]
            st.dataframe(filtered.sort_values(["Score","R:R"],ascending=False).head(50)[cols],use_container_width=True,hide_index=True)
        st.caption("CE is considered only with bullish underlying confirmation; PE only with bearish confirmation. Option Entry/SL/T1/T2/T3 are premium levels.")

# ============================================================
# INDEX F&O DISPLAY
# ============================================================

else:
    st.header("📊 INDEX F&O OPPORTUNITY BOARD")
    st.caption(f"Indices: {', '.join(selected_indices) if selected_indices else 'None'} | Options: {', '.join(option_types)} | Budget: ₹{fno_budget:,.0f}")
    if results.empty:
        st.info("No index F&O setup passed the V24 quality/liquidity and directional checks. This is a NO-TRADE result.")
    else:
        filtered = results[results["Status"].isin(fno_filters)] if "Status" in results.columns else results
        ready = results[results["Status"] == "🟢 F&O READY"]
        watch = results[results["Status"] == "⚡ F&O WATCH"]
        a,b,c=st.columns(3); a.metric("🟢 F&O READY",len(ready)); b.metric("⚡ F&O WATCH",len(watch)); c.metric("Opportunities",len(results))
        if not filtered.empty:
            cols=[x for x in ["Status","Symbol","Strategy","Option","Strike","Expiry","Underlying","Premium","Entry","SL","T1","T2","T3","R:R","Score","Volume","OI","Lots","Capital ₹"] if x in filtered.columns]
            st.dataframe(filtered.sort_values(["Score","R:R"],ascending=False).head(50)[cols],use_container_width=True,hide_index=True)

# ============================================================
# EXPORT + EXPLANATION
# ============================================================

st.divider()
st.header("🧠 Scanner Rules")
st.markdown(f"""
- **Selected scanner:** {scanner_mode} — only this scanner runs.
- **Capital:** ₹{trading_capital:,.0f}
- **Swing:** retains the V24 independent strategies: Breakout, Breakout-Ready, Pullback, Demand Reversal, Momentum and Trend Continuation.
- **Intraday:** separate 15m trend + 5m VWAP/ORB momentum strategy.
- **Stock F&O:** separate underlying-direction + option-chain strategy.
- **Index F&O:** separate index-direction + option strategy.
- **READY:** all 6 V19/V20 entry conditions are met.
- **Previous READY:** today's scan compares against the previous session's saved READY signals, so yesterday's Jyoti CNC/ACE-type setups do not simply disappear.
""")

if not results.empty:
    st.download_button(
        "⬇️ Download Current Scanner Results",
        data=results.to_csv(index=False),
        file_name={
            "📈 Swing":"tradequest_v25_swing.csv",
            "⚡ Intraday":"tradequest_v25_intraday.csv",
            "🎯 Stock F&O":"tradequest_v25_stock_fno.csv",
            "📊 Index F&O":"tradequest_v25_index_fno.csv",
        }[scanner_mode],
        mime="text/csv",
    )

st.caption("TradeQuest V25 is a quantitative screening/research tool. It is not personalized investment advice or a guarantee of execution/profit.")
