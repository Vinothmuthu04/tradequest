
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# TRADEQUEST V20
# ============================================================
#
# V20 = V19 STRATEGY + STOCK DECISION ANALYZER
#
# TARGET        = 8%
# STOP          = 5%
# RISK          = 1%
# MAX POSITIONS = 5
# MAX HOLD      = 20 DAYS
#
# ENTRY:
#   Price > MA20
#   MA20 > MA50
#   RSI 50-70
#   MACD > Signal
#   Volume >= 1.5x 20D average
#   50D breakout
#
# DECISION ANALYZER:
#   BUY NOW
#   NEAR ENTRY / WAIT
#   AVERAGE / HOLD
#   DO NOT BUY
#
# ============================================================


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TradeQuest V20",
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

PORTFOLIO_ALLOCATION = (
    INITIAL_CAPITAL / MAX_POSITIONS
)


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36",

    "Accept":
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8",

    "Accept-Language":
        "en-US,en;q=0.9",

    "Referer":
        "https://www.nseindia.com/"
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
# CLEAN SYMBOL
# ============================================================

def clean_symbol(symbol):

    if symbol is None:
        return None

    symbol = str(symbol).strip().upper()

    if not symbol:
        return None

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

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
# FULL NSE EQUITY LIST
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
            "NSE full equity list could not be downloaded.\n\n"
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
            "SYMBOL column not found in NSE file."
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
        .str.upper()
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
            f"Only {len(stocks)} usable NSE symbols found."
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

        info = ticker.info

        market_cap = info.get(
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
            "No stocks above ₹10,000 Cr found."
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
# DOWNLOAD DATA
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

        if not all(
            col in data.columns
            for col in required
        ):

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

    # MA
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

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

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
    ema12 = (
        close.ewm(
            span=12,
            adjust=False
        ).mean()
    )

    ema26 = (
        close.ewm(
            span=26,
            adjust=False
        ).mean()
    )

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

            close = float(row["Close"])
            dma200 = float(row["DMA200"])

            if np.isnan(close):
                continue

            if np.isnan(dma200):
                continue

            total += 1

            if close > dma200:
                above += 1

        except Exception:

            continue

    if total == 0:
        return True

    breadth = (
        above / total
    ) * 100

    return breadth >= 50


# ============================================================
# CREATE SIGNALS
# ============================================================

def create_signals(
    symbol,
    data,
    priority
):

    signals = []

    if data is None:
        return signals

    for i in range(220, len(data)):

        row = data.iloc[i]

        if not signal_passes(row):
            continue

        entry = float(row["Close"])

        stop = (
            entry *
            (1 - STOP_PERCENT / 100)
        )

        target = (
            entry *
            (1 + TARGET_PERCENT / 100)
        )

        max_exit_index = min(
            i + MAX_HOLDING_DAYS,
            len(data) - 1
        )

        signals.append({

            "symbol": symbol,
            "entry_index": i,
            "entry_date": data.index[i],
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "max_exit_index": max_exit_index,
            "priority": priority,
            "technical_score": technical_score(row),
            "data": data

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

        all_signals.extend(signals)

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

    cash = float(INITIAL_CAPITAL)

    positions = []

    trades = []

    equity_history = []

    for current_date in all_dates:

        # ----------------------------------------------------
        # EXITS
        # ----------------------------------------------------

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

            row = data.iloc[current_index]

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
                exit_value - investment
            )

            costs = (
                investment *
                ROUND_TRIP_COST_PERCENT /
                100
            )

            net_pnl = gross_pnl - costs

            cash += (
                investment + net_pnl
            )

            trades.append({

                "Symbol": position["symbol"],
                "Entry Date": position["entry_date"],
                "Exit Date": current_date,
                "Entry": round(entry, 2),
                "Exit": round(exit_price, 2),
                "Shares": shares,
                "Universe Priority": position["priority"],
                "Technical Score": position["technical_score"],
                "Investment": round(investment, 2),
                "P&L": round(net_pnl, 2),
                "Return %": round(
                    net_pnl / investment * 100,
                    2
                ),
                "Reason": reason

            })

        positions = remaining

        # ----------------------------------------------------
        # NEW ENTRIES
        # ----------------------------------------------------

        slots = (
            MAX_POSITIONS -
            len(positions)
        )

        if slots > 0:

            market_ok = True

            if use_market_filter:

                market_ok = market_filter_passes(
                    current_date,
                    stock_data
                )

            if market_ok:

                todays_signals = [

                    x for x in all_signals

                    if x["entry_date"]
                    == current_date

                ]

                todays_signals.sort(

                    key=lambda x: (

                        x["priority"],
                        x["technical_score"],
                        float(
                            x["data"]
                            .iloc[x["entry_index"]]
                            ["VOLUME_RATIO"]
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

                        "symbol": signal["symbol"],
                        "entry_date": signal["entry_date"],
                        "entry_index": signal["entry_index"],
                        "entry_price": entry_price,
                        "stop_price": signal["stop_price"],
                        "target_price": signal["target_price"],
                        "max_exit_index": signal["max_exit_index"],
                        "shares": shares,
                        "priority": signal["priority"],
                        "technical_score": signal["technical_score"],
                        "data": signal["data"]

                    })

        # ----------------------------------------------------
        # EQUITY
        # ----------------------------------------------------

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

                    price = position["entry_price"]

            except Exception:

                price = position["entry_price"]

            portfolio_value += (
                position["shares"] *
                price
            )

        equity_history.append({

            "Date": current_date,
            "Equity": portfolio_value

        })

    # --------------------------------------------------------
    # CLOSE OPEN POSITIONS
    # --------------------------------------------------------

    if all_dates:

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

                exit_price = position["entry_price"]

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

            net_pnl = gross_pnl - costs

            cash += (
                investment +
                net_pnl
            )

            trades.append({

                "Symbol": position["symbol"],
                "Entry Date": position["entry_date"],
                "Exit Date": final_date,
                "Entry": round(entry, 2),
                "Exit": round(exit_price, 2),
                "Shares": shares,
                "Universe Priority": position["priority"],
                "Technical Score": position["technical_score"],
                "Investment": round(investment, 2),
                "P&L": round(net_pnl, 2),
                "Return %": round(
                    net_pnl / investment * 100,
                    2
                ),
                "Reason": "END OF TEST"

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

    if equity is None or equity.empty:
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
        )
        /
        peak
    ) * 100

    return {

        "Final Capital": final_capital,
        "Return %": total_return,
        "Trades": total_trades,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,

        "Avg Trade %":
            trades["Return %"].mean(),

        "Avg Winner %":
            wins["Return %"].mean()
            if not wins.empty
            else 0,

        "Avg Loser %":
            losses["Return %"].mean()
            if not losses.empty
            else 0,

        "Max Drawdown %":
            drawdown.min(),

        "Best Trade %":
            trades["Return %"].max(),

        "Worst Trade %":
            trades["Return %"].min()

    }


# ============================================================
# CURRENT CANDIDATES
# ============================================================

def scan_latest_candidates(
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

        if not signal_passes(row):
            continue

        priority = priorities.get(
            symbol,
            0
        )

        membership = []

        if symbol in nifty200_symbols:
            membership.append("NIFTY 200")

        if symbol in nifty500_symbols:
            membership.append("NIFTY 500")

        if symbol in marketcap_symbols:
            membership.append("₹10,000 Cr+")

        results.append({

            "Symbol": symbol,

            "Price": round(
                float(row["Close"]),
                2
            ),

            "Priority": priority,

            "Priority Level":
                priority_label(priority),

            "Technical Score":
                technical_score(row),

            "RSI": round(
                float(row["RSI"]),
                1
            ),

            "Volume Ratio": round(
                float(row["VOLUME_RATIO"]),
                2
            ),

            "MA20": round(
                float(row["MA20"]),
                2
            ),

            "MA50": round(
                float(row["MA50"]),
                2
            ),

            "Breakout": "YES",

            "Membership":
                " + ".join(membership)

        })

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)

    result = result.sort_values(
        by=[
            "Priority",
            "Technical Score",
            "Volume Ratio"
        ],
        ascending=False
    )

    return result.reset_index(drop=True)


# ============================================================
# NEAR ENTRY SCANNER
# ============================================================

def scan_near_entry_candidates(
    stock_data,
    priorities
):

    results = []

    for symbol, data in stock_data.items():

        if data is None or data.empty:
            continue

        row = data.iloc[-1]

        try:

            price = float(row["Close"])
            ma20 = float(row["MA20"])
            ma50 = float(row["MA50"])
            rsi = float(row["RSI"])
            macd = float(row["MACD"])
            signal = float(row["MACD_SIGNAL"])
            volume = float(row["VOLUME_RATIO"])
            breakout = bool(row["BREAKOUT50"])

        except Exception:

            continue

        conditions = {

            "Price > MA20":
                price > ma20,

            "MA20 > MA50":
                ma20 > ma50,

            "RSI 50-70":
                50 <= rsi <= 70,

            "MACD > Signal":
                macd > signal,

            "Volume >= 1.5x":
                volume >= 1.5,

            "50D Breakout":
                breakout

        }

        passed = sum(
            conditions.values()
        )

        # Near entry = at least 4 of 6
        # and trend is not broken.

        if (
            passed >= 4
            and
            price > ma50
        ):

            missing = [
                name
                for name, value
                in conditions.items()
                if not value
            ]

            results.append({

                "Symbol": symbol,

                "Price": round(
                    price,
                    2
                ),

                "Priority":
                    priorities.get(
                        symbol,
                        0
                    ),

                "Technical Score":
                    technical_score(row),

                "Conditions":
                    f"{passed}/6",

                "RSI":
                    round(
                        rsi,
                        1
                    ),

                "Volume":
                    round(
                        volume,
                        2
                    ),

                "MA20":
                    round(
                        ma20,
                        2
                    ),

                "MA50":
                    round(
                        ma50,
                        2
                    ),

                "Missing":
                    ", ".join(missing)
                    if missing
                    else "None",

                "Entry":
                    round(
                        price,
                        2
                    ),

                "Stop":
                    round(
                        price *
                        (
                            1 -
                            STOP_PERCENT /
                            100
                        ),
                        2
                    ),

                "Target":
                    round(
                        price *
                        (
                            1 +
                            TARGET_PERCENT /
                            100
                        ),
                        2
                    )

            })

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)

    result = result.sort_values(
        by=[
            "Conditions",
            "Priority",
            "Technical Score"
        ],
        ascending=False
    )

    return result.reset_index(drop=True)


# ============================================================
# PORTFOLIO PLAN
# ============================================================

def build_portfolio_plan(
    candidates,
    capital=INITIAL_CAPITAL
):

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    selected = candidates.head(
        MAX_POSITIONS
    )

    allocation = (
        capital /
        MAX_POSITIONS
    )

    rows = []

    for _, row in selected.iterrows():

        price = float(
            row["Price"]
        )

        if price <= 0:
            continue

        shares = int(
            allocation /
            price
        )

        if shares <= 0:
            continue

        investment = shares * price

        stop = (
            price *
            (
                1 -
                STOP_PERCENT /
                100
            )
        )

        target = (
            price *
            (
                1 +
                TARGET_PERCENT /
                100
            )
        )

        risk = (
            shares *
            (price - stop)
        )

        profit = (
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

            "Entry":
                round(price, 2),

            "Shares":
                shares,

            "Investment ₹":
                round(investment),

            "Stop ₹":
                round(stop, 2),

            "Target ₹":
                round(target, 2),

            "Risk ₹":
                round(risk),

            "Potential Profit ₹":
                round(profit),

            "RSI":
                row["RSI"],

            "Volume":
                row["Volume Ratio"]

        })

    return pd.DataFrame(rows)


# ============================================================
# SINGLE STOCK ANALYZER
# ============================================================

def analyze_single_stock(
    symbol,
    existing_buy_price=0.0
):

    symbol = clean_symbol(symbol)

    if not symbol:

        return {
            "error":
                "Invalid stock symbol."
        }

    data = download_stock_data(
        symbol
    )

    if data is None or data.empty:

        return {
            "error":
                f"No valid data found for {symbol}."
        }

    row = data.iloc[-1]

    try:

        price = float(row["Close"])
        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])
        rsi = float(row["RSI"])
        macd = float(row["MACD"])
        macd_signal = float(
            row["MACD_SIGNAL"]
        )
        volume = float(
            row["VOLUME_RATIO"]
        )
        breakout = bool(
            row["BREAKOUT50"]
        )

    except Exception:

        return {
            "error":
                "Technical indicators unavailable."
        }

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    c1 = price > ma20
    c2 = ma20 > ma50
    c3 = 50 <= rsi <= 70
    c4 = macd > macd_signal
    c5 = volume >= 1.5
    c6 = breakout

    passed = sum([
        c1, c2, c3, c4, c5, c6
    ])

    score = technical_score(
        row
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    try:

        n200, n500 = (
            build_universe_memberships()
        )

    except Exception:

        n200 = set()
        n500 = set()

    priority = 0

    if symbol in n200:
        priority += 5

    if symbol in n500:
        priority += 3

    # --------------------------------------------------------
    # PRICE PLAN
    # --------------------------------------------------------

    entry = price

    stop = (
        entry *
        (
            1 -
            STOP_PERCENT /
            100
        )
    )

    target = (
        entry *
        (
            1 +
            TARGET_PERCENT /
            100
        )
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if passed == 6:

        decision = "🟢 BUY NOW"

        reason = (
            "All 6 V20 entry conditions are satisfied."
        )

    elif (
        passed >= 4
        and
        price > ma20
        and
        ma20 > ma50
    ):

        decision = "🟡 NEAR ENTRY / WAIT"

        reason = (
            f"{passed}/6 conditions passed. "
            "Wait for the missing confirmation."
        )

    elif (
        existing_buy_price > 0
        and
        price > ma50
        and
        ma20 > ma50
        and
        rsi >= 45
        and
        macd > macd_signal
    ):

        decision = "🔵 AVERAGE / HOLD"

        reason = (
            "Existing position remains "
            "technically healthy. "
            "Do not average blindly."
        )

    else:

        decision = "🔴 DO NOT BUY"

        reason = (
            f"Only {passed}/6 conditions passed. "
            "The V20 setup is not strong enough."
        )

    # --------------------------------------------------------
    # POSITION SIZE
    # --------------------------------------------------------

    shares = int(
        PORTFOLIO_ALLOCATION /
        price
    )

    investment = (
        shares *
        price
    )

    risk = (
        shares *
        (price - stop)
    )

    potential_profit = (
        shares *
        (target - price)
    )

    # --------------------------------------------------------
    # EXISTING POSITION
    # --------------------------------------------------------

    existing_return = None

    if existing_buy_price > 0:

        existing_return = (
            (
                price -
                existing_buy_price
            )
            /
            existing_buy_price
        ) * 100

    return {

        "Symbol":
            symbol,

        "Decision":
            decision,

        "Reason":
            reason,

        "Current Price":
            price,

        "Entry":
            entry,

        "Stop Loss":
            stop,

        "Target":
            target,

        "Technical Score":
            score,

        "Conditions":
            f"{passed}/6",

        "RSI":
            rsi,

        "MA20":
            ma20,

        "MA50":
            ma50,

        "MACD":
            macd,

        "MACD Signal":
            macd_signal,

        "Volume Ratio":
            volume,

        "Breakout":
            breakout,

        "Priority":
            priority,

        "Suggested Shares":
            shares,

        "Investment":
            investment,

        "Risk":
            risk,

        "Potential Profit":
            potential_profit,

        "Existing Buy Price":
            existing_buy_price,

        "Existing Return":
            existing_return

    }


# ============================================================
# YEARLY RESULTS
# ============================================================

def yearly_results(trades):

    if trades is None or trades.empty:
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
                round(
                    pf,
                    2
                ),

            "P&L ₹":
                round(
                    group["P&L"].sum(),
                    2
                )

        })

    return pd.DataFrame(rows)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 TradeQuest V20"
)

st.subheader(
    "NSE Swing Trading Scanner + "
    "Backtest + Stock Decision Analyzer"
)

st.info(
    "V20 uses the locked V20 rules: "
    "8% target, 5% stop, 1% risk, "
    "maximum 5 positions and 20-day maximum hold."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ V20 Settings"
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
    "### Entry Conditions"
)

st.sidebar.write("✓ Price > MA20")
st.sidebar.write("✓ MA20 > MA50")
st.sidebar.write("✓ RSI 50-70")
st.sidebar.write("✓ MACD > Signal")
st.sidebar.write("✓ Volume ≥ 1.5x")
st.sidebar.write("✓ 50D Breakout")


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Universe",
    universe
)

c2.metric(
    "Market Filter",
    "ON" if use_market_filter else "OFF"
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
# STOCK DECISION ANALYZER
# ============================================================

st.header(
    "🔍 Stock Decision Analyzer"
)

st.write(
    "Enter any NSE stock to check whether "
    "the V20 rules say BUY NOW, "
    "NEAR ENTRY / WAIT, AVERAGE / HOLD, "
    "or DO NOT BUY."
)

a1, a2, a3 = st.columns(3)

with a1:

    analyzer_symbol = st.text_input(
        "📌 Stock Symbol",
        placeholder="Example: APOLLOMICRO"
    )

with a2:

    existing_buy_price = st.number_input(
        "💰 Existing Buy Price",
        min_value=0.0,
        value=0.0,
        step=0.05
    )

with a3:

    analyze_button = st.button(
        "🔎 ANALYZE",
        type="primary"
    )


if analyze_button:

    if not analyzer_symbol.strip():

        st.warning(
            "Enter a stock symbol."
        )

    else:

        with st.spinner(
            f"Analyzing {analyzer_symbol.upper()}..."
        ):

            analysis = analyze_single_stock(
                analyzer_symbol,
                existing_buy_price
            )

        if "error" in analysis:

            st.error(
                analysis["error"]
            )

        else:

            decision = analysis["Decision"]

            if "BUY NOW" in decision:

                st.success(
                    f"## {decision}"
                )

            elif "NEAR ENTRY" in decision:

                st.warning(
                    f"## {decision}"
                )

            elif "AVERAGE" in decision:

                st.info(
                    f"## {decision}"
                )

            else:

                st.error(
                    f"## {decision}"
                )

            st.write(
                f"**{analysis['Symbol']}** — "
                f"{analysis['Reason']}"
            )

            # ------------------------------------------------
            # ENTRY / SL / TARGET
            # ------------------------------------------------

            st.subheader(
                "🎯 Entry / Stop Loss / Target"
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Current Price",
                f"₹{analysis['Current Price']:.2f}"
            )

            p2.metric(
                "Entry",
                f"₹{analysis['Entry']:.2f}"
            )

            p3.metric(
                "Stop Loss",
                f"₹{analysis['Stop Loss']:.2f}"
            )

            p4.metric(
                "Target",
                f"₹{analysis['Target']:.2f}"
            )

            # ------------------------------------------------
            # TECHNICALS
            # ------------------------------------------------

            st.subheader(
                "📊 Technical Strength"
            )

            t1, t2, t3, t4, t5 = st.columns(5)

            t1.metric(
                "Score",
                analysis["Technical Score"]
            )

            t2.metric(
                "Conditions",
                analysis["Conditions"]
            )

            t3.metric(
                "RSI",
                f"{analysis['RSI']:.1f}"
            )

            t4.metric(
                "Volume",
                f"{analysis['Volume Ratio']:.2f}x"
            )

            t5.metric(
                "Priority",
                analysis["Priority"]
            )

            technical_df = pd.DataFrame({

                "Condition": [

                    "Price > MA20",
                    "MA20 > MA50",
                    "RSI 50-70",
                    "MACD > Signal",
                    "Volume >= 1.5x",
                    "50D Breakout"

                ],

                "Status": [

                    "✅ PASS"
                    if analysis["Current Price"]
                    >
                    analysis["MA20"]
                    else "❌ FAIL",

                    "✅ PASS"
                    if analysis["MA20"]
                    >
                    analysis["MA50"]
                    else "❌ FAIL",

                    "✅ PASS"
                    if 50 <= analysis["RSI"] <= 70
                    else "❌ FAIL",

                    "✅ PASS"
                    if analysis["MACD"]
                    >
                    analysis["MACD Signal"]
                    else "❌ FAIL",

                    "✅ PASS"
                    if analysis["Volume Ratio"] >= 1.5
                    else "❌ FAIL",

                    "✅ PASS"
                    if analysis["Breakout"]
                    else "❌ FAIL"

                ]

            })

            st.dataframe(
                technical_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # POSITION PLAN
            # ------------------------------------------------

            st.subheader(
                "💰 ₹5 Lakh Position Plan"
            )

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                "Shares",
                analysis["Suggested Shares"]
            )

            s2.metric(
                "Investment",
                f"₹{analysis['Investment']:,.0f}"
            )

            s3.metric(
                "Risk",
                f"₹{analysis['Risk']:,.0f}"
            )

            s4.metric(
                "Potential Profit",
                f"₹{analysis['Potential Profit']:,.0f}"
            )

            # ------------------------------------------------
            # EXISTING POSITION
            # ------------------------------------------------

            if analysis["Existing Buy Price"] > 0:

                st.subheader(
                    "📦 Existing Position"
                )

                if analysis["Existing Return"] >= 0:

                    st.success(
                        f"Existing return: "
                        f"{analysis['Existing Return']:.2f}%"
                    )

                else:

                    st.error(
                        f"Existing return: "
                        f"{analysis['Existing Return']:.2f}%"
                    )

                st.caption(
                    "AVERAGE / HOLD is not an instruction "
                    "to blindly add more shares. "
                    "Risk management remains mandatory."
                )


# ============================================================
# RUN BACKTEST
# ============================================================

st.header(
    "🚀 V20 Backtest"
)

run_button = st.button(
    "🚀 RUN V20 BACKTEST",
    type="primary"
)

if run_button:

    # --------------------------------------------------------
    # MEMBERSHIPS
    # --------------------------------------------------------

    with st.spinner(
        "Loading NIFTY 200 / NIFTY 500..."
    ):

        (
            nifty200_symbols,
            nifty500_symbols
        ) = build_universe_memberships()

    # --------------------------------------------------------
    # MARKET CAP
    # --------------------------------------------------------

    marketcap_stocks = []

    if universe in [
        "All Stocks > ₹10,000 Cr",
        "Combined"
    ]:

        st.warning(
            "Building the genuine NSE-wide "
            "₹10,000 Cr+ universe. "
            "This can take time."
        )

        try:

            marketcap_stocks = (
                build_market_cap_universe()
            )

        except Exception as e:

            st.error(
                str(e)
            )

            st.stop()

    marketcap_symbols = {
        x["symbol"]
        for x in marketcap_stocks
    }

    # --------------------------------------------------------
    # SELECT UNIVERSE
    # --------------------------------------------------------

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
            nifty200_symbols
            |
            nifty500_symbols
            |
            marketcap_symbols
        )

    if not selected_symbols:

        st.error(
            "Selected universe is empty."
        )

        st.stop()

    # --------------------------------------------------------
    # PRIORITIES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DOWNLOAD HISTORY
    # --------------------------------------------------------

    st.header(
        "📥 Historical Data"
    )

    stock_data = {}

    progress = st.progress(0)

    status = st.empty()

    symbols = sorted(
        selected_symbols
    )

    total = len(symbols)

    successful = 0
    skipped = 0

    for i, symbol in enumerate(symbols):

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

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    with st.spinner(
        "Running 5-year V20 backtest..."
    ):

        trades, equity = run_backtest(
            stock_data,
            priorities,
            use_market_filter
        )

    if trades is None or trades.empty:

        st.error(
            "No trades generated."
        )

        st.stop()

    metrics = calculate_metrics(
        trades,
        equity
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.header(
        "🏆 V20 Full 5-Year Result"
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

    # --------------------------------------------------------
    # TRADE QUALITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CURRENT CANDIDATES
    # --------------------------------------------------------

    st.header(
        "🔎 Current Swing Candidates"
    )

    candidates = scan_latest_candidates(

        stock_data,

        priorities,

        (
            nifty200_symbols,
            nifty500_symbols,
            marketcap_symbols
        )

    )

    if candidates.empty:

        st.info(
            "No stocks currently satisfy "
            "all 6 V20 entry conditions."
        )

    else:

        st.dataframe(
            candidates,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PORTFOLIO
        # ----------------------------------------------------

        st.header(
            "💰 ₹5 Lakh Portfolio Plan"
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

            total_profit = (
                portfolio["Potential Profit ₹"].sum()
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

            st.dataframe(
                portfolio,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Potential gross profit at 8% target: "
                f"₹{total_profit:,.0f}"
            )

    # --------------------------------------------------------
    # NEAR ENTRY
    # --------------------------------------------------------

    st.header(
        "🟡 Near-Entry Stocks"
    )

    near_entry = scan_near_entry_candidates(
        stock_data,
        priorities
    )

    if near_entry.empty:

        st.info(
            "No strong near-entry setups currently."
        )

    else:

        st.write(
            "These stocks are not full V20 entries yet, "
            "but their technical structure is developing."
        )

        st.dataframe(
            near_entry,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    peak = equity["Equity"].cummax()

    drawdown = (
        (
            equity["Equity"] -
            peak
        )
        /
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

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(

        "⬇️ Download Trades CSV",

        data=trades.to_csv(
            index=False
        ),

        file_name=(
            "tradequest_v20_"
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
    "TradeQuest V20 | "
    "Backtesting and research tool only. "
    "Past performance does not guarantee "
    "future results."
)

