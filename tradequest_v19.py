import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# TRADEQUEST V19
# ============================================================
#
# V19 = V18 + LIVE NEAR-ENTRY SCANNER
#
# LOCKED STRATEGY
# ------------------------------------------------------------
# TARGET        = 8%
# STOP          = 5%
# RISK          = 1%
# MAX POSITIONS = 5
# MAX HOLD      = 20 DAYS
#
# ENTRY CONDITIONS
# ------------------------------------------------------------
# 1. Price > MA20
# 2. MA20 > MA50
# 3. RSI 50-70
# 4. MACD > Signal
# 5. Volume >= 1.5x 20D average
# 6. 50D breakout
#
# V19 LIVE STATES
# ------------------------------------------------------------
# 🟢 READY       = 6/6 conditions
# 🟡 NEAR ENTRY  = 5/6 conditions
# 🟠 WATCH       = 3-4/6 conditions
# ⚪ WEAK        = below 3/6
#
# NEAR-ENTRY SHOWS
# ------------------------------------------------------------
# Current Price
# Entry Trigger
# Distance to Entry
# Stop Loss
# Target
# Priority
# Technical Score
# RSI
# Volume Ratio
# Missing Conditions
#
# BACKTEST STRATEGY IS UNCHANGED
#
# ============================================================


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TradeQuest V19",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

INITIAL_CAPITAL = 500000

TARGET_PERCENT = 8.0
STOP_PERCENT = 5.0
RISK_PERCENT = 1.0

MAX_POSITIONS = 5
MAX_HOLDING_DAYS = 20

VOLUME_MULTIPLIER = 1.5

MARKET_CAP_CRORES = 10000
MARKET_CAP_RUPEES = MARKET_CAP_CRORES * 10_000_000

HISTORY_PERIOD = "5y"

ROUND_TRIP_COST_PERCENT = 0.30


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}


# ============================================================
# URLS
# ============================================================

NSE_SECURITY_LIST_URL = (
    "https://nsearchives.nseindia.com/"
    "content/equities/EQUITY_L.csv"
)

NIFTY_200_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty200list.csv"
)

NIFTY_500_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty500list.csv"
)


# ============================================================
# SESSION
# ============================================================

def create_nse_session():

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get(
            "https://www.nseindia.com/",
            timeout=20
        )
    except Exception:
        pass

    return session


# ============================================================
# SYMBOL CLEANING
# ============================================================

def clean_symbol(symbol):

    if symbol is None:
        return None

    symbol = str(symbol).strip()

    if not symbol:
        return None

    return symbol


# ============================================================
# NIFTY LIST
# ============================================================

@st.cache_data(ttl=3600)
def load_nifty_list(universe):

    url = (
        NIFTY_200_URL
        if universe == "NIFTY 200"
        else NIFTY_500_URL
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        df = pd.read_csv(
            io.StringIO(response.text)
        )

        stocks = []

        for _, row in df.iterrows():

            symbol = clean_symbol(
                row.get("Symbol")
            )

            if symbol:

                stocks.append({
                    "symbol": symbol,
                    "company": str(
                        row.get(
                            "Company Name",
                            symbol
                        )
                    ).strip()
                })

        return stocks

    except Exception as e:

        st.error(
            f"Could not download {universe} list."
        )
        st.error(str(e))

        return []


# ============================================================
# FULL NSE LIST
# ============================================================

@st.cache_data(ttl=86400)
def load_full_nse_equity_list():

    session = create_nse_session()

    try:

        response = session.get(
            NSE_SECURITY_LIST_URL,
            timeout=30
        )

        response.raise_for_status()

        df = pd.read_csv(
            io.StringIO(response.text)
        )

    except Exception as e:

        raise RuntimeError(
            "NSE full equity security list "
            "could not be downloaded.\n\n"
            "No NIFTY 500 fallback is used.\n\n"
            f"Error: {e}"
        )

    symbol_column = None

    for column in [
        "SYMBOL",
        "Symbol",
        "symbol"
    ]:

        if column in df.columns:
            symbol_column = column
            break

    if symbol_column is None:

        raise RuntimeError(
            "NSE file was downloaded but "
            "SYMBOL column was not found."
        )

    series_column = None

    for column in [
        " SERIES",
        "SERIES",
        "Series",
        "series"
    ]:

        if column in df.columns:
            series_column = column
            break

    if series_column:

        df[series_column] = (
            df[series_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df = df[
            df[series_column] == "EQ"
        ]

    df[symbol_column] = (
        df[symbol_column]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[symbol_column] != ""
    ]

    df = df.drop_duplicates(
        subset=[symbol_column]
    )

    stocks = []

    for _, row in df.iterrows():

        symbol = clean_symbol(
            row[symbol_column]
        )

        if symbol:

            stocks.append({
                "symbol": symbol,
                "company": symbol
            })

    if len(stocks) < 500:

        raise RuntimeError(
            f"NSE equity list contains only "
            f"{len(stocks)} usable symbols."
        )

    return stocks


# ============================================================
# MARKET CAP
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_market_cap(symbol):

    try:

        ticker = yf.Ticker(
            symbol + ".NS"
        )

        market_cap = ticker.info.get(
            "marketCap"
        )

        if market_cap is None:
            return None

        return float(market_cap)

    except Exception:

        return None


# ============================================================
# MARKET CAP UNIVERSE
# ============================================================

def build_market_cap_universe():

    stocks = load_full_nse_equity_list()

    eligible = []

    progress = st.progress(0)
    status = st.empty()

    total = len(stocks)

    for i, stock in enumerate(stocks):

        symbol = stock["symbol"]

        status.text(
            f"Checking market cap "
            f"{i + 1}/{total}: {symbol}"
        )

        market_cap = get_market_cap(
            symbol
        )

        if (
            market_cap is not None
            and market_cap >= MARKET_CAP_RUPEES
        ):

            eligible.append({
                "symbol": symbol,
                "company": stock["company"],
                "market_cap": market_cap
            })

        progress.progress(
            (i + 1) / total
        )

    progress.empty()
    status.empty()

    if not eligible:

        raise RuntimeError(
            "No stocks above ₹10,000 Cr were found."
        )

    return eligible


# ============================================================
# MEMBERSHIPS
# ============================================================

def build_universe_memberships():

    nifty200 = load_nifty_list(
        "NIFTY 200"
    )

    nifty500 = load_nifty_list(
        "NIFTY 500"
    )

    nifty200_symbols = {
        x["symbol"]
        for x in nifty200
    }

    nifty500_symbols = {
        x["symbol"]
        for x in nifty500
    }

    return (
        nifty200_symbols,
        nifty500_symbols
    )


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(
    symbol,
    nifty200_symbols,
    nifty500_symbols,
    marketcap_symbols
):

    score = 0

    if symbol in nifty200_symbols:
        score += 5

    if symbol in nifty500_symbols:
        score += 3

    if symbol in marketcap_symbols:
        score += 2

    return score


def priority_label(priority):

    if priority >= 10:
        return "🔟 TOP PRIORITY"

    if priority >= 8:
        return "8️⃣ HIGH PRIORITY"

    if priority >= 5:
        return "5️⃣ GOOD"

    if priority >= 3:
        return "3️⃣ WATCH"

    if priority >= 2:
        return "2️⃣ WATCH"

    return "0️⃣ LOW"


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def download_stock_data(symbol):

    try:

        data = yf.download(
            symbol + ".NS",
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data is None or data.empty:
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in required:

            if column not in data.columns:
                return None

        data = data.dropna(
            subset=required
        )

        if len(data) < 250:
            return None

        return calculate_indicators(
            data
        )

    except Exception:

        return None


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(data):

    data = data.copy()

    close = data["Close"]
    high = data["High"]
    volume = data["Volume"]

    data["MA20"] = (
        close.rolling(20).mean()
    )

    data["MA50"] = (
        close.rolling(50).mean()
    )

    data["DMA200"] = (
        close.rolling(200).mean()
    )

    # RSI

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain.rolling(14).mean()
    )

    avg_loss = (
        loss.rolling(14).mean()
    )

    rs = (
        avg_gain / avg_loss
    )

    data["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # MACD

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # Volume

    data["AVG_VOLUME20"] = (
        volume.rolling(20).mean()
    )

    data["VOLUME_RATIO"] = (
        volume /
        data["AVG_VOLUME20"]
    )

    # 50D breakout

    data["RESISTANCE50"] = (
        high
        .rolling(50)
        .max()
        .shift(1)
    )

    data["BREAKOUT50"] = (
        close >
        data["RESISTANCE50"]
    )

    return data


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(row):

    try:

        close = float(row["Close"])
        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])
        rsi = float(row["RSI"])
        macd = float(row["MACD"])
        signal = float(row["MACD_SIGNAL"])
        volume_ratio = float(
            row["VOLUME_RATIO"]
        )

        breakout = bool(
            row["BREAKOUT50"]
        )

    except Exception:

        return 0

    score = 0

    if close > ma20:
        score += 20

    if ma20 > ma50:
        score += 20

    if 50 <= rsi <= 70:
        score += 15

    elif 45 <= rsi < 50:
        score += 8

    if macd > signal:
        score += 15

    if volume_ratio >= 2.0:
        score += 20

    elif volume_ratio >= 1.5:
        score += 15

    elif volume_ratio >= 1.2:
        score += 8

    if breakout:
        score += 10

    return min(score, 100)


# ============================================================
# STRICT ENTRY
# ============================================================

def signal_passes(row):

    try:

        close = float(row["Close"])
        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])
        rsi = float(row["RSI"])
        macd = float(row["MACD"])
        signal = float(row["MACD_SIGNAL"])
        volume_ratio = float(
            row["VOLUME_RATIO"]
        )

        breakout = bool(
            row["BREAKOUT50"]
        )

    except Exception:

        return False

    return (
        close > ma20
        and ma20 > ma50
        and 50 <= rsi <= 70
        and macd > signal
        and volume_ratio >= VOLUME_MULTIPLIER
        and breakout
    )


# ============================================================
# CONDITION ANALYSIS
# ============================================================

def get_conditions(row):

    try:

        close = float(row["Close"])
        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])
        rsi = float(row["RSI"])
        macd = float(row["MACD"])
        signal = float(row["MACD_SIGNAL"])
        volume_ratio = float(
            row["VOLUME_RATIO"]
        )

        breakout = bool(
            row["BREAKOUT50"]
        )

    except Exception:

        return {
            "Price > MA20": False,
            "MA20 > MA50": False,
            "RSI 50-70": False,
            "MACD > Signal": False,
            "Volume >= 1.5x": False,
            "50D Breakout": False
        }

    return {

        "Price > MA20":
            close > ma20,

        "MA20 > MA50":
            ma20 > ma50,

        "RSI 50-70":
            50 <= rsi <= 70,

        "MACD > Signal":
            macd > signal,

        "Volume >= 1.5x":
            volume_ratio >= VOLUME_MULTIPLIER,

        "50D Breakout":
            breakout
    }


# ============================================================
# MISSING CONDITIONS
# ============================================================

def missing_conditions(row):

    conditions = get_conditions(row)

    missing = [
        name
        for name, passed
        in conditions.items()
        if not passed
    ]

    return missing


# ============================================================
# NEAR ENTRY STATE
# ============================================================

def get_setup_state(row):

    conditions = get_conditions(row)

    passed = sum(
        conditions.values()
    )

    if passed == 6:
        return "🟢 READY", passed

    if passed == 5:
        return "🟡 NEAR ENTRY", passed

    if passed >= 3:
        return "🟠 WATCH", passed

    return "⚪ WEAK", passed


# ============================================================
# ENTRY TRIGGER
# ============================================================

def calculate_entry_trigger(row):

    try:

        close = float(row["Close"])
        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])
        resistance = float(
            row["RESISTANCE50"]
        )

    except Exception:

        return None

    candidates = [
        close,
        ma20,
        ma50,
        resistance
    ]

    candidates = [
        x for x in candidates
        if np.isfinite(x)
    ]

    if not candidates:
        return None

    #
    # For a developing setup, use the
    # highest structural trigger.
    #
    # This prevents calling a stock READY
    # before the actual breakout.
    #

    return max(candidates)


# ============================================================
# LIVE CANDIDATE SCANNER
# ============================================================

def scan_live_candidates(
    stock_data,
    priorities,
    memberships
):

    nifty200_symbols = memberships[0]
    nifty500_symbols = memberships[1]
    marketcap_symbols = memberships[2]

    results = []

    for symbol, data in stock_data.items():

        if data is None or data.empty:
            continue

        row = data.iloc[-1]

        try:

            price = float(
                row["Close"]
            )

            ma20 = float(
                row["MA20"]
            )

            ma50 = float(
                row["MA50"]
            )

            rsi = float(
                row["RSI"]
            )

            volume_ratio = float(
                row["VOLUME_RATIO"]
            )

            macd = float(
                row["MACD"]
            )

            macd_signal = float(
                row["MACD_SIGNAL"]
            )

            resistance = float(
                row["RESISTANCE50"]
            )

        except Exception:

            continue

        if not all(
            np.isfinite(x)
            for x in [
                price,
                ma20,
                ma50,
                rsi,
                volume_ratio,
                macd,
                macd_signal,
                resistance
            ]
        ):

            continue

        state, passed = get_setup_state(
            row
        )

        priority = priorities.get(
            symbol,
            0
        )

        trigger = calculate_entry_trigger(
            row
        )

        if trigger is None:
            continue

        #
        # For READY stocks the current price
        # is the entry.
        #
        # For developing stocks the structural
        # trigger is displayed.
        #

        if passed == 6:

            entry = price

        else:

            entry = max(
                price,
                trigger
            )

        stop = (
            entry *
            (
                1 -
                STOP_PERCENT / 100
            )
        )

        target = (
            entry *
            (
                1 +
                TARGET_PERCENT / 100
            )
        )

        distance = (
            (
                entry -
                price
            )
            /
            price
            *
            100
        )

        missing = missing_conditions(
            row
        )

        membership = []

        if symbol in nifty200_symbols:
            membership.append(
                "NIFTY 200"
            )

        if symbol in nifty500_symbols:
            membership.append(
                "NIFTY 500"
            )

        if symbol in marketcap_symbols:
            membership.append(
                "₹10,000 Cr+"
            )

        results.append({

            "Status":
                state,

            "Passed":
                f"{passed}/6",

            "Symbol":
                symbol,

            "Current ₹":
                round(price, 2),

            "Entry Trigger ₹":
                round(entry, 2),

            "Distance %":
                round(distance, 2),

            "Stop Loss ₹":
                round(stop, 2),

            "Target ₹":
                round(target, 2),

            "Priority":
                priority,

            "Priority Level":
                priority_label(priority),

            "Technical Score":
                technical_score(row),

            "RSI":
                round(rsi, 1),

            "Volume Ratio":
                round(volume_ratio, 2),

            "MA20":
                round(ma20, 2),

            "MA50":
                round(ma50, 2),

            "50D Resistance":
                round(resistance, 2),

            "Missing Conditions":
                ", ".join(missing)
                if missing
                else "NONE",

            "Membership":
                " + ".join(membership)

        })

    if not results:

        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    state_order = {
        "🟢 READY": 0,
        "🟡 NEAR ENTRY": 1,
        "🟠 WATCH": 2,
        "⚪ WEAK": 3
    }

    result["_state_order"] = (
        result["Status"]
        .map(state_order)
        .fillna(9)
    )

    result = result.sort_values(
        by=[
            "_state_order",
            "Priority",
            "Technical Score",
            "Distance %"
        ],
        ascending=[
            True,
            False,
            False,
            True
        ]
    )

    result.drop(
        columns=["_state_order"],
        inplace=True
    )

    result.reset_index(
        drop=True,
        inplace=True
    )

    return result


# ============================================================
# READY / NEAR ENTRY TABLE
# ============================================================

def display_live_scanner(
    candidates
):

    if candidates.empty:

        st.info(
            "No stocks currently have "
            "a usable V19 setup."
        )

        return

    ready = candidates[
        candidates["Status"] ==
        "🟢 READY"
    ]

    near = candidates[
        candidates["Status"] ==
        "🟡 NEAR ENTRY"
    ]

    watch = candidates[
        candidates["Status"] ==
        "🟠 WATCH"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🟢 READY",
        len(ready)
    )

    c2.metric(
        "🟡 NEAR ENTRY",
        len(near)
    )

    c3.metric(
        "🟠 WATCH",
        len(watch)
    )

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    if not ready.empty:

        st.subheader(
            "🟢 READY — Entry Conditions Met"
        )

        st.dataframe(
            ready,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # NEAR ENTRY
    # --------------------------------------------------------

    if not near.empty:

        st.subheader(
            "🟡 NEAR ENTRY — Watch Closely"
        )

        st.info(
            "These stocks satisfy 5/6 V19 "
            "conditions. The missing condition "
            "must become valid before entry."
        )

        st.dataframe(
            near,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    if not watch.empty:

        st.subheader(
            "🟠 WATCH — Setup Developing"
        )

        st.dataframe(
            watch,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PORTFOLIO PLAN
# ============================================================

def build_portfolio_plan(
    candidates,
    capital=INITIAL_CAPITAL
):

    if candidates.empty:
        return pd.DataFrame()

    #
    # ONLY READY STOCKS
    #
    selected = candidates[
        candidates["Status"] ==
        "🟢 READY"
    ].head(
        MAX_POSITIONS
    ).copy()

    if selected.empty:
        return pd.DataFrame()

    allocation = (
        capital /
        MAX_POSITIONS
    )

    rows = []

    for _, row in selected.iterrows():

        price = float(
            row["Entry Trigger ₹"]
        )

        if price <= 0:
            continue

        shares = int(
            allocation /
            price
        )

        if shares <= 0:
            continue

        investment = (
            shares *
            price
        )

        stop = (
            price *
            (
                1 -
                STOP_PERCENT / 100
            )
        )

        target = (
            price *
            (
                1 +
                TARGET_PERCENT / 100
            )
        )

        risk = (
            shares *
            (price - stop)
        )

        potential = (
            shares *
            (target - price)
        )

        rows.append({

            "Rank":
                len(rows) + 1,

            "Symbol":
                row["Symbol"],

            "Priority":
                row["Priority"],

            "Technical Score":
                row["Technical Score"],

            "Entry ₹":
                round(price, 2),

            "Shares":
                shares,

            "Investment ₹":
                round(investment, 0),

            "Stop Loss ₹":
                round(stop, 2),

            "Target ₹":
                round(target, 2),

            "Risk ₹":
                round(risk, 0),

            "Potential Profit ₹":
                round(potential, 0),

            "RSI":
                row["RSI"],

            "Volume":
                row["Volume Ratio"]

        })

    return pd.DataFrame(rows)


# ============================================================
# MARKET FILTER
# ============================================================

def market_filter_passes(
    current_date,
    stock_data
):

    total = 0
    above = 0

    for data in stock_data.values():

        if data is None:
            continue

        try:

            available = data.index[
                data.index <= current_date
            ]

            if len(available) == 0:
                continue

            date = available[-1]

            row = data.loc[date]

            close = float(
                row["Close"]
            )

            dma200 = float(
                row["DMA200"]
            )

            if not np.isfinite(close):
                continue

            if not np.isfinite(dma200):
                continue

            total += 1

            if close > dma200:
                above += 1

        except Exception:

            continue

    if total == 0:
        return True

    breadth = (
        above /
        total *
        100
    )

    return breadth >= 50


# ============================================================
# CREATE BACKTEST SIGNALS
# ============================================================

def create_signals(
    symbol,
    data,
    priority
):

    signals = []

    if data is None:
        return signals

    for i in range(
        220,
        len(data)
    ):

        row = data.iloc[i]

        if not signal_passes(row):
            continue

        entry_price = float(
            row["Close"]
        )

        stop_price = (
            entry_price *
            (
                1 -
                STOP_PERCENT / 100
            )
        )

        target_price = (
            entry_price *
            (
                1 +
                TARGET_PERCENT / 100
            )
        )

        max_exit_index = min(
            i + MAX_HOLDING_DAYS,
            len(data) - 1
        )

        signals.append({

            "symbol":
                symbol,

            "entry_index":
                i,

            "entry_date":
                data.index[i],

            "entry_price":
                entry_price,

            "stop_price":
                stop_price,

            "target_price":
                target_price,

            "max_exit_index":
                max_exit_index,

            "priority":
                priority,

            "technical_score":
                technical_score(row),

            "data":
                data

        })

    return signals


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    stock_data,
    priorities,
    use_market_filter
):

    all_signals = []

    for symbol, data in stock_data.items():

        signals = create_signals(
            symbol,
            data,
            priorities.get(symbol, 0)
        )

        all_signals.extend(
            signals
        )

    if not all_signals:
        return None, None

    all_signals.sort(
        key=lambda x: x["entry_date"]
    )

    all_dates = sorted(
        set(
            date
            for data in stock_data.values()
            if data is not None
            for date in data.index
        )
    )

    cash = float(
        INITIAL_CAPITAL
    )

    positions = []
    trades = []
    equity_history = []

    for current_date in all_dates:

        # ====================================================
        # EXITS
        # ====================================================

        remaining = []

        for position in positions:

            data = position["data"]

            try:

                available = data.index[
                    data.index <= current_date
                ]

                if len(available) == 0:
                    remaining.append(position)
                    continue

                actual_date = available[-1]

                current_index = (
                    data.index.get_loc(
                        actual_date
                    )
                )

            except Exception:

                remaining.append(position)
                continue

            if current_index <= position["entry_index"]:

                remaining.append(position)
                continue

            row = data.iloc[
                current_index
            ]

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            exit_price = None
            reason = None

            if low <= position["stop_price"]:

                exit_price = position["stop_price"]
                reason = "STOP LOSS"

            elif high >= position["target_price"]:

                exit_price = position["target_price"]
                reason = "TARGET"

            elif current_index >= position["max_exit_index"]:

                exit_price = close
                reason = "TIME EXIT"

            if exit_price is None:

                remaining.append(position)
                continue

            shares = position["shares"]
            entry = position["entry_price"]

            investment = shares * entry
            exit_value = shares * exit_price

            gross_pnl = (
                exit_value -
                investment
            )

            costs = (
                investment *
                ROUND_TRIP_COST_PERCENT /
                100
            )

            net_pnl = (
                gross_pnl -
                costs
            )

            cash += (
                investment +
                net_pnl
            )

            trades.append({

                "Symbol":
                    position["symbol"],

                "Entry Date":
                    position["entry_date"],

                "Exit Date":
                    current_date,

                "Entry":
                    round(entry, 2),

                "Exit":
                    round(exit_price, 2),

                "Shares":
                    shares,

                "Universe Priority":
                    position["priority"],

                "Technical Score":
                    position["technical_score"],

                "Investment":
                    round(investment, 2),

                "P&L":
                    round(net_pnl, 2),

                "Return %":
                    round(
                        net_pnl /
                        investment *
                        100,
                        2
                    ),

                "Reason":
                    reason

            })

        positions = remaining

        # ====================================================
        # ENTRIES
        # ====================================================

        slots = (
            MAX_POSITIONS -
            len(positions)
        )

        if slots > 0:

            market_ok = True

            if use_market_filter:

                market_ok = (
                    market_filter_passes(
                        current_date,
                        stock_data
                    )
                )

            if market_ok:

                todays_signals = [
                    signal
                    for signal in all_signals
                    if signal["entry_date"]
                    == current_date
                ]

                todays_signals.sort(
                    key=lambda x: (
                        x["priority"],
                        x["technical_score"],
                        float(
                            x["data"]
                            .iloc[
                                x["entry_index"]
                            ]["VOLUME_RATIO"]
                        )
                    ),
                    reverse=True
                )

                for signal in todays_signals[:slots]:

                    entry_price = signal[
                        "entry_price"
                    ]

                    risk_amount = (
                        INITIAL_CAPITAL *
                        RISK_PERCENT /
                        100
                    )

                    risk_per_share = (
                        entry_price -
                        signal["stop_price"]
                    )

                    if risk_per_share <= 0:
                        continue

                    shares_by_risk = int(
                        risk_amount /
                        risk_per_share
                    )

                    shares_by_cash = int(
                        cash /
                        entry_price
                    )

                    shares = min(
                        shares_by_risk,
                        shares_by_cash
                    )

                    if shares <= 0:
                        continue

                    investment = (
                        shares *
                        entry_price
                    )

                    cash -= investment

                    positions.append({

                        "symbol":
                            signal["symbol"],

                        "entry_date":
                            signal["entry_date"],

                        "entry_index":
                            signal["entry_index"],

                        "entry_price":
                            entry_price,

                        "stop_price":
                            signal["stop_price"],

                        "target_price":
                            signal["target_price"],

                        "max_exit_index":
                            signal["max_exit_index"],

                        "shares":
                            shares,

                        "priority":
                            signal["priority"],

                        "technical_score":
                            signal["technical_score"],

                        "data":
                            signal["data"]

                    })

        # ====================================================
        # EQUITY
        # ====================================================

        portfolio_value = cash

        for position in positions:

            data = position["data"]

            try:

                available = data.index[
                    data.index <= current_date
                ]

                if len(available) > 0:

                    actual_date = available[-1]

                    price = float(
                        data.loc[
                            actual_date,
                            "Close"
                        ]
                    )

                else:

                    price = position[
                        "entry_price"
                    ]

            except Exception:

                price = position[
                    "entry_price"
                ]

            portfolio_value += (
                position["shares"] *
                price
            )

        equity_history.append({

            "Date":
                current_date,

            "Equity":
                portfolio_value

        })

    # ========================================================
    # CLOSE OPEN POSITIONS
    # ========================================================

    final_date = all_dates[-1]

    for position in positions:

        data = position["data"]

        try:

            available = data.index[
                data.index <= final_date
            ]

            actual_date = available[-1]

            exit_price = float(
                data.loc[
                    actual_date,
                    "Close"
                ]
            )

        except Exception:

            exit_price = position[
                "entry_price"
            ]

        shares = position["shares"]
        entry = position["entry_price"]

        investment = shares * entry
        exit_value = shares * exit_price

        gross_pnl = (
            exit_value -
            investment
        )

        costs = (
            investment *
            ROUND_TRIP_COST_PERCENT /
            100
        )

        net_pnl = (
            gross_pnl -
            costs
        )

        cash += (
            investment +
            net_pnl
        )

        trades.append({

            "Symbol":
                position["symbol"],

            "Entry Date":
                position["entry_date"],

            "Exit Date":
                final_date,

            "Entry":
                round(entry, 2),

            "Exit":
                round(exit_price, 2),

            "Shares":
                shares,

            "Universe Priority":
                position["priority"],

            "Technical Score":
                position["technical_score"],

            "Investment":
                round(investment, 2),

            "P&L":
                round(net_pnl, 2),

            "Return %":
                round(
                    net_pnl /
                    investment *
                    100,
                    2
                ),

            "Reason":
                "END OF TEST"

        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_history)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
    equity
):

    if trades is None or trades.empty:
        return None

    final_capital = float(
        equity["Equity"].iloc[-1]
    )

    total_return = (
        final_capital /
        INITIAL_CAPITAL -
        1
    ) * 100

    wins = trades[
        trades["P&L"] > 0
    ]

    losses = trades[
        trades["P&L"] <= 0
    ]

    total_trades = len(trades)

    win_rate = (
        len(wins) /
        total_trades *
        100
        if total_trades
        else 0
    )

    gross_profit = wins["P&L"].sum()

    gross_loss = abs(
        losses["P&L"].sum()
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else 999
    )

    peak = equity["Equity"].cummax()

    drawdown = (
        (
            equity["Equity"] -
            peak
        ) /
        peak
    ) * 100

    return {

        "Final Capital":
            final_capital,

        "Return %":
            total_return,

        "Trades":
            total_trades,

        "Win Rate %":
            win_rate,

        "Profit Factor":
            profit_factor,

        "Avg Trade %":
            trades["Return %"].mean(),

        "Avg Winner %":
            wins["Return %"].mean()
            if not wins.empty else 0,

        "Avg Loser %":
            losses["Return %"].mean()
            if not losses.empty else 0,

        "Max Drawdown %":
            drawdown.min(),

        "Best Trade %":
            trades["Return %"].max(),

        "Worst Trade %":
            trades["Return %"].min()

    }


# ============================================================
# YEARLY RESULTS
# ============================================================

def yearly_results(trades):

    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()

    df["Year"] = (
        pd.to_datetime(
            df["Exit Date"]
        ).dt.year
    )

    rows = []

    for year, group in df.groupby(
        "Year"
    ):

        wins = group[
            group["P&L"] > 0
        ]

        losses = group[
            group["P&L"] <= 0
        ]

        gp = wins["P&L"].sum()

        gl = abs(
            losses["P&L"].sum()
        )

        pf = (
            gp / gl
            if gl > 0
            else 999
        )

        rows.append({

            "Year":
                year,

            "Trades":
                len(group),

            "Win Rate %":
                round(
                    len(wins) /
                    len(group) *
                    100,
                    2
                ),

            "Profit Factor":
                round(pf, 2),

            "P&L ₹":
                round(
                    group["P&L"].sum(),
                    2
                )

        })

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================

st.title(
    "📈 TradeQuest V19"
)

st.subheader(
    "Multi-Universe Swing Scanner + "
    "Near-Entry Detection + "
    "₹5 Lakh Portfolio Planner"
)

st.info(
    "V19 keeps the V17/V18 strategy locked "
    "and adds a live setup-state scanner. "
    "READY means 6/6 conditions. "
    "NEAR ENTRY means 5/6 conditions."
)

st.sidebar.header(
    "⚙️ V19 Settings"
)

universe = st.sidebar.selectbox(
    "📊 Stock Universe",
    [
        "NIFTY 200",
        "NIFTY 500",
        "All Stocks > ₹10,000 Cr",
        "Combined"
    ]
)

use_market_filter = st.sidebar.checkbox(
    "🌐 Market Filter",
    value=False
)

st.sidebar.divider()

st.sidebar.write(
    "### 🔒 Locked Strategy"
)

st.sidebar.write("🎯 Target: 8%")
st.sidebar.write("🛑 Stop: 5%")
st.sidebar.write("💰 Risk: 1%")
st.sidebar.write("📌 Max Positions: 5")
st.sidebar.write("💵 Capital: ₹5,00,000")
st.sidebar.write("⏳ Max Hold: 20 days")

st.sidebar.divider()

st.sidebar.write(
    "### Entry"
)

st.sidebar.write("✓ Price > MA20")
st.sidebar.write("✓ MA20 > MA50")
st.sidebar.write("✓ RSI 50–70")
st.sidebar.write("✓ MACD > Signal")
st.sidebar.write("✓ Volume ≥ 1.5×")
st.sidebar.write("✓ 50D Breakout")


# ============================================================
# HEADER METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Universe",
    universe
)

c2.metric(
    "Market Filter",
    "ON"
    if use_market_filter
    else "OFF"
)

c3.metric(
    "Target",
    "8%"
)

c4.metric(
    "Stop",
    "5%"
)


# ============================================================
# RUN
# ============================================================

if st.button(
    "🚀 RUN V19",
    type="primary"
):

    # ========================================================
    # MEMBERSHIPS
    # ========================================================

    with st.spinner(
        "Loading NIFTY 200 / NIFTY 500..."
    ):

        (
            nifty200_symbols,
            nifty500_symbols
        ) = build_universe_memberships()

    # ========================================================
    # MARKET CAP
    # ========================================================

    marketcap_stocks = []

    if universe in [
        "All Stocks > ₹10,000 Cr",
        "Combined"
    ]:

        st.warning(
            "Building the genuine NSE-wide "
            "₹10,000 Cr+ universe. "
            "This can take time because "
            "market cap is checked across "
            "the NSE equity list."
        )

        try:

            marketcap_stocks = (
                build_market_cap_universe()
            )

        except Exception as e:

            st.error(
                "❌ Could not build "
                "₹10,000 Cr+ universe."
            )

            st.error(str(e))

            st.stop()

    marketcap_symbols = {
        stock["symbol"]
        for stock in marketcap_stocks
    }

    # ========================================================
    # SELECT UNIVERSE
    # ========================================================

    if universe == "NIFTY 200":

        selected_symbols = (
            nifty200_symbols
        )

    elif universe == "NIFTY 500":

        selected_symbols = (
            nifty500_symbols
        )

    elif universe == "All Stocks > ₹10,000 Cr":

        selected_symbols = (
            marketcap_symbols
        )

    else:

        selected_symbols = (
            nifty200_symbols |
            nifty500_symbols |
            marketcap_symbols
        )

    if not selected_symbols:

        st.error(
            "Selected universe is empty."
        )

        st.stop()

    # ========================================================
    # PRIORITIES
    # ========================================================

    priorities = {}

    for symbol in selected_symbols:

        priorities[symbol] = (
            calculate_priority(
                symbol,
                nifty200_symbols,
                nifty500_symbols,
                marketcap_symbols
            )
        )

    # ========================================================
    # HISTORICAL DATA
    # ========================================================

    st.header(
        "📥 Historical Data"
    )

    stock_data = {}

    progress = st.progress(0)
    status = st.empty()

    total = len(selected_symbols)

    successful = 0
    skipped = 0

    for i, symbol in enumerate(
        sorted(selected_symbols)
    ):

        status.text(
            f"Downloading "
            f"{i + 1}/{total}: "
            f"{symbol}"
        )

        data = download_stock_data(
            symbol
        )

        if data is not None:

            stock_data[symbol] = data
            successful += 1

        else:

            skipped += 1

        progress.progress(
            (i + 1) / total
        )

    progress.empty()
    status.empty()

    st.success(
        f"Downloaded: {successful} | "
        f"Skipped: {skipped}"
    )

    if successful < 20:

        st.error(
            "Too few valid stocks."
        )

        st.stop()

    # ========================================================
    # UNIVERSE SUMMARY
    # ========================================================

    st.header(
        "🌐 Universe Summary"
    )

    u1, u2, u3, u4 = st.columns(4)

    u1.metric(
        "NIFTY 200",
        len(nifty200_symbols)
    )

    u2.metric(
        "NIFTY 500",
        len(nifty500_symbols)
    )

    u3.metric(
        "₹10,000 Cr+",
        len(marketcap_symbols)
    )

    u4.metric(
        "Selected",
        len(selected_symbols)
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    with st.spinner(
        "Running 5-year V19 backtest..."
    ):

        trades, equity = run_backtest(
            stock_data,
            priorities,
            use_market_filter
        )

    if trades is None:

        st.error(
            "No trades generated."
        )

        st.stop()

    metrics = calculate_metrics(
        trades,
        equity
    )

    # ========================================================
    # RESULTS
    # ========================================================

    st.header(
        "🏆 V19 Full 5-Year Result"
    )

    st.caption(
        f"Universe: {universe} | "
        f"Market Filter: "
        f"{'ON' if use_market_filter else 'OFF'} | "
        f"Target: 8% | Stop: 5%"
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Final Capital",
        f"₹{metrics['Final Capital']:,.0f}"
    )

    r2.metric(
        "Return",
        f"{metrics['Return %']:.2f}%"
    )

    r3.metric(
        "Profit Factor",
        f"{metrics['Profit Factor']:.2f}"
    )

    r4.metric(
        "Max Drawdown",
        f"{metrics['Max Drawdown %']:.2f}%"
    )

    r5, r6, r7, r8 = st.columns(4)

    r5.metric(
        "Trades",
        metrics["Trades"]
    )

    r6.metric(
        "Win Rate",
        f"{metrics['Win Rate %']:.2f}%"
    )

    r7.metric(
        "Avg Trade",
        f"{metrics['Avg Trade %']:.2f}%"
    )

    r8.metric(
        "Worst Trade",
        f"{metrics['Worst Trade %']:.2f}%"
    )

    # ========================================================
    # TRADE QUALITY
    # ========================================================

    st.subheader(
        "📊 Trade Quality"
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Average Winner",
        f"{metrics['Avg Winner %']:.2f}%"
    )

    q2.metric(
        "Average Loser",
        f"{metrics['Avg Loser %']:.2f}%"
    )

    q3.metric(
        "Best Trade",
        f"{metrics['Best Trade %']:.2f}%"
    )

    q4.metric(
        "Stocks Tested",
        successful
    )

    # ========================================================
    # PRIORITY
    # ========================================================

    st.header(
        "🏆 Universe Priority"
    )

    priority_counts = (
        pd.Series(
            list(
                priorities.values()
            )
        )
        .value_counts()
        .sort_index(
            ascending=False
        )
    )

    priority_df = pd.DataFrame({

        "Priority":
            priority_counts.index,

        "Stocks":
            priority_counts.values

    })

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # LIVE SCANNER
    # ========================================================

    st.header(
        "🔎 V19 Live Entry Scanner"
    )

    st.caption(
        "READY = 6/6 conditions | "
        "NEAR ENTRY = 5/6 | "
        "WATCH = 3–4/6"
    )

    memberships = (
        nifty200_symbols,
        nifty500_symbols,
        marketcap_symbols
    )

    candidates = scan_live_candidates(
        stock_data,
        priorities,
        memberships
    )

    display_live_scanner(
        candidates
    )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    ready_candidates = candidates[
        candidates["Status"] ==
        "🟢 READY"
    ] if not candidates.empty else pd.DataFrame()

    if not ready_candidates.empty:

        st.header(
            "💰 V19 ₹5 Lakh Portfolio Plan"
        )

        st.info(
            "Only READY stocks are included "
            "in the illustrative portfolio. "
            "NEAR ENTRY stocks are watchlist "
            "candidates and are NOT treated "
            "as active entries."
        )

        portfolio = build_portfolio_plan(
            candidates,
            INITIAL_CAPITAL
        )

        if not portfolio.empty:

            total_investment = (
                portfolio["Investment ₹"].sum()
            )

            total_risk = (
                portfolio["Risk ₹"].sum()
            )

            total_potential = (
                portfolio[
                    "Potential Profit ₹"
                ].sum()
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Capital",
                "₹5,00,000"
            )

            p2.metric(
                "Positions",
                len(portfolio)
            )

            p3.metric(
                "Capital Used",
                f"₹{total_investment:,.0f}"
            )

            p4.metric(
                "Planned Risk",
                f"₹{total_risk:,.0f}"
            )

            st.subheader(
                "🥇 TOP READY ACTION LIST"
            )

            st.dataframe(
                portfolio,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Potential gross profit if "
                f"all selected stocks reach "
                f"the 8% target: "
                f"₹{total_potential:,.0f}"
            )

    else:

        st.info(
            "No READY stocks currently. "
            "Use the 🟡 NEAR ENTRY section "
            "as the watchlist."
        )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    st.header(
        "📈 Equity Curve"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity["Date"],
            y=equity["Equity"],
            mode="lines",
            name="Portfolio"
        )
    )

    fig.add_hline(
        y=INITIAL_CAPITAL,
        line_dash="dot"
    )

    fig.update_layout(
        height=500,
        title=(
            f"{universe} | "
            f"8% Target | "
            f"5% Stop"
        ),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ₹",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    peak = equity["Equity"].cummax()

    drawdown = (
        (
            equity["Equity"] -
            peak
        ) /
        peak
    ) * 100

    st.header(
        "📉 Drawdown"
    )

    dd_fig = go.Figure()

    dd_fig.add_trace(
        go.Scatter(
            x=equity["Date"],
            y=drawdown,
            mode="lines",
            name="Drawdown"
        )
    )

    dd_fig.update_layout(
        height=400,
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        template="plotly_white"
    )

    st.plotly_chart(
        dd_fig,
        use_container_width=True
    )

    # ========================================================
    # YEARLY RESULTS
    # ========================================================

    st.header(
        "📅 Year-by-Year Results"
    )

    yearly = yearly_results(
        trades
    )

    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ALL TRADES
    # ========================================================

    st.header(
        "📋 All Trades"
    )

    st.dataframe(
        trades.sort_values(
            "Entry Date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        "⬇️ Download Trades CSV",
        data=trades.to_csv(
            index=False
        ),
        file_name=(
            "tradequest_v19_"
            +
            universe.replace(
                " ",
                "_"
            )
            +
            ".csv"
        ),
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TradeQuest V19 | "
    "Backtesting and research tool only. "
    "Past performance does not guarantee "
    "future results."
)