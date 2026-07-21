import time
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, datetime

st.set_page_config(page_title="Stock Scanner", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
    --bg: #0B0F17; --panel: #111726; --line: #1C2536;
    --text: #D9E1EC; --muted: #8A94A6; --accent: #E8A13D;
    --buy: #2FBF8F; --short: #E5484D;
}
.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; font-size: 1.55rem !important;
    letter-spacing: 0.12em; color: var(--text) !important;
    border-bottom: 1px solid var(--line); padding-bottom: 0.8rem !important;
}
h1::before { content: ""; display: inline-block; width: 0.5em; height: 0.85em; background: var(--accent); margin-right: 0.5em; }
.status {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    color: var(--muted); background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px;
    padding: 10px 16px; margin: 4px 0 14px 0; letter-spacing: 0.04em;
}
.status b { color: var(--text); font-weight: 500; }
.status .buy { color: var(--buy); }
.status .short { color: var(--short); }
.status .sep { color: var(--line); padding: 0 10px; }
.status .live { color: var(--accent); }
.alertline {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    color: var(--text); background: var(--panel);
    border: 1px solid var(--line); border-left: 3px solid var(--accent);
    border-radius: 6px; padding: 10px 16px; margin: 0 0 18px 0;
}
.alertline .buy { color: var(--buy); font-weight: 600; }
.alertline .short { color: var(--short); font-weight: 600; }
[data-testid="stSidebar"] { background: var(--bg); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace; }
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.05em; color: var(--muted);
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent); }
[data-testid="stDataFrame"], .stSelectbox > div, .stTextArea textarea { border-radius: 6px; }
.stCaption, small { color: var(--muted) !important; }
hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Watchlist")
DEFAULT = ("AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,"
           "SPY,QQQ,PLTR,COIN,SOFI,DIS,BA,UBER,SHOP,INTC,"
           "ES=F,MES=F,NQ=F,MNQ=F,YM=F,RTY=F")
tickers = [t.strip().upper() for t in
           st.sidebar.text_area("Tickers (comma separated)", DEFAULT, height=140).split(",") if t.strip()]
period = st.sidebar.selectbox("History period", ["3mo", "6mo", "1y"], index=1)
rsi_buy = st.sidebar.slider("RSI oversold (buy)", 10, 40, 30)
rsi_short = st.sidebar.slider("RSI overbought (short)", 60, 90, 70)
live = st.sidebar.toggle("Live refresh (60s)", value=False)


def daily_signals(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    std20 = close.rolling(20).std()
    bb_up, bb_lo = sma20 + 2 * std20, sma20 - 2 * std20
    score = ((rsi < rsi_buy).astype(int) * 2
             - (rsi > rsi_short).astype(int) * 2
             + (macd > macd_sig).astype(int) * 2 - 1
             + (sma20 > sma50).astype(int) * 2 - 1
             + (close <= bb_lo).astype(int)
             - (close >= bb_up).astype(int))
    sig = pd.Series("HOLD", index=close.index)
    sig[score >= 2] = "BUY"
    sig[score <= -2] = "SHORT"
    valid = sma50.notna()
    return (sig[valid], close[valid],
            {"rsi": rsi.iloc[-1], "sma20": sma20.iloc[-1], "sma50": sma50.iloc[-1],
             "macd": macd.iloc[-1], "macd_signal": macd_sig.iloc[-1],
             "bb_upper": bb_up.iloc[-1], "bb_lower": bb_lo.iloc[-1]})


def reasons_text(r, close_last):
    why = []
    if r["rsi"] < rsi_buy: why.append(f"RSI {r['rsi']:.0f} oversold")
    elif r["rsi"] > rsi_short: why.append(f"RSI {r['rsi']:.0f} overbought")
    why.append("MACD bullish" if r["macd"] > r["macd_signal"] else "MACD bearish")
    why.append("Uptrend" if r["sma20"] > r["sma50"] else "Downtrend")
    if close_last <= r["bb_lower"]: why.append("Lower Bollinger")
    elif close_last >= r["bb_upper"]: why.append("Upper Bollinger")
    return " | ".join(why)


def signal_age(sig, close):
    current = sig.iloc[-1]
    flip_pos = 0
    for i in range(len(sig) - 1, -1, -1):
        if sig.iloc[i] != current:
            flip_pos = i + 1
            break
    days = len(sig) - flip_pos
    capped = flip_pos == 0
    entry = float(close.iloc[flip_pos])
    since = sig.index[flip_pos]
    chg = (float(close.iloc[-1]) / entry - 1) * 100
    days_str = f"{days}+" if capped else str(days)
    return days_str, f"{since:%m-%d}", entry, chg


def backtest(sig, close):
    """Long while BUY, short while SHORT, flat on HOLD; act next day."""
    pos = sig.map({"BUY": 1, "SHORT": -1, "HOLD": 0}).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strat = float((1 + pos * ret).prod() - 1)
    bh = float(close.iloc[-1] / close.iloc[0] - 1)
    trades, cur, entry_i = [], 0, None
    vals = pos.tolist()
    for i, p in enumerate(vals):
        if p != cur:
            if cur != 0 and entry_i is not None:
                trades.append((float(close.iloc[i - 1]) / float(close.iloc[entry_i]) - 1) * cur)
            cur, entry_i = p, (i if p != 0 else None)
    if cur != 0 and entry_i is not None:
        trades.append((float(close.iloc[-1]) / float(close.iloc[entry_i]) - 1) * cur)
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades) * 100 if trades else 0.0
    return strat * 100, bh * 100, len(trades), win_rate


@st.cache_data(ttl=900, show_spinner=False)
def load(tickers, period, bucket):
    return yf.download(tickers, period=period, interval="1d",
                       progress=False, auto_adjust=True, group_by="ticker")


st.title("STOCK SCANNER")
if not tickers:
    st.info("Add a ticker in the sidebar."); st.stop()

bucket = int(time.time() // 60) if live else 0
with st.spinner(f"Scanning {len(tickers)} tickers..."):
    raw = load(tickers, period, bucket)
if raw is None:
    st.error("No data returned."); st.stop()

rows, bt_rows, failed = [], [], []
for t in tickers:
    try:
        close_full = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
        if len(close_full) < 60: failed.append(t); continue
        sig_series, close_v, r = daily_signals(close_full)
        if sig_series.empty: failed.append(t); continue
        s = sig_series.iloc[-1]
        days, since, entry, chg_since = signal_age(sig_series, close_v)
        rows.append({
            "ticker": t, "signal": s, "days": days, "since": since,
            "entry": entry, "chg_since_%": chg_since,
            "close": float(close_v.iloc[-1]),
            "chg_1d_%": (float(close_v.iloc[-1]) / float(close_v.iloc[-2]) - 1) * 100,
            "rsi": r["rsi"],
            "reasons": reasons_text(r, float(close_v.iloc[-1])),
        })
        strat, bh, ntr, wr = backtest(sig_series, close_v)
        bt_rows.append({"ticker": t, "strategy_%": strat, "buy_hold_%": bh,
                        "edge_%": strat - bh, "trades": ntr, "win_rate_%": wr})
    except Exception:
        failed.append(t)

if failed: st.sidebar.warning("No data for: " + ", ".join(failed))
if not rows: st.error("No data returned."); st.stop()

df = pd.DataFrame(rows)
bt = pd.DataFrame(bt_rows)
b = int((df["signal"] == "BUY").sum())
s = int((df["signal"] == "SHORT").sum())
h = len(df) - b - s

live_txt = (f'<span class="sep">|</span><span class="live">LIVE '
            f'{datetime.now():%H:%M:%S}</span>') if live else ""
st.markdown(
    f'<div class="status">{date.today():%a %d %b %Y}'
    f'<span class="sep">|</span><b>{len(df)}</b> scanned'
    f'<span class="sep">|</span><span class="buy">{b} buy</span>'
    f'<span class="sep">|</span><span class="short">{s} short</span>'
    f'<span class="sep">|</span>{h} hold{live_txt}</div>',
    unsafe_allow_html=True,
)

alerts = df[df["signal"] != "HOLD"]
if not alerts.empty:
    parts = [f'<span class="{str(r.signal).lower()}">{r.signal}</span> {r.ticker} ({r.days}d)'
             for r in alerts.itertuples()]
    st.markdown('<div class="alertline">' + '&nbsp;&nbsp;&middot;&nbsp;&nbsp;'.join(parts) + '</div>',
                unsafe_allow_html=True)
    if not live:
        st.toast(f"{len(alerts)} setup(s) found")
        st.markdown(
            '<audio autoplay><source '
            'src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" '
            'type="audio/ogg"></audio>',
            unsafe_allow_html=True,
        )


def cs(v):
    if v == "BUY": return "color:#2FBF8F;font-weight:600"
    if v == "SHORT": return "color:#E5484D;font-weight:600"
    return "color:#8A94A6"


def cc(v):
    try: return "color:#2FBF8F" if v >= 0 else "color:#E5484D"
    except TypeError: return ""


cols = ["ticker", "signal", "days", "since", "entry", "chg_since_%",
        "close", "chg_1d_%", "rsi", "reasons"]


def styled(d):
    return (d[cols].style
            .map(cs, subset=["signal"])
            .map(cc, subset=["chg_since_%", "chg_1d_%"])
            .format(precision=2))


t1, t2, t3 = st.tabs(["ALL TICKERS", "SETUPS", "BACKTEST"])
with t1: st.dataframe(styled(df), use_container_width=True, height=560)
with t2:
    st_df = df[df["signal"] != "HOLD"]
    if st_df.empty: st.info("No buy/short setups right now.")
    else: st.dataframe(styled(st_df), use_container_width=True, height=400)
with t3:
    bt_sorted = bt.sort_values("edge_%", ascending=False)
    st.dataframe(
        bt_sorted.style
        .map(cc, subset=["strategy_%", "buy_hold_%", "edge_%"])
        .format(precision=1),
        use_container_width=True, height=520,
    )
    avg_edge = bt["edge_%"].mean()
    beat = int((bt["edge_%"] > 0).sum())
    st.caption(
        f"Following every signal over the loaded {period} beat buy-and-hold on "
        f"{beat} of {len(bt)} tickers (avg edge {avg_edge:+.1f}%). "
        "Simulated on daily closes, no fees or slippage. Past results do not predict future ones."
    )

st.divider()
pick = st.selectbox("Chart", df["ticker"])
close_c = raw[pick]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
st.line_chart(
    pd.DataFrame({"Close": close_c,
                  "SMA 20": close_c.rolling(20).mean(),
                  "SMA 50": close_c.rolling(50).mean()}),
    height=380,
    color=["#E8A13D", "#5B6B85", "#33415C"],
)
row = df[df["ticker"] == pick].iloc[0]
st.markdown(f"`{pick}`  {row['signal']} since {row['since']}  |  "
            f"{row['days']} sessions  |  {row['chg_since_%']:+.1f}% since signal  |  {row['reasons']}")
st.caption("Simple indicator rules (RSI, MACD, SMA, Bollinger). Screening ideas, not financial advice.")
