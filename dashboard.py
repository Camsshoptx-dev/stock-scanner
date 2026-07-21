import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, datetime

st.set_page_config(page_title="Stock Scanner", layout="wide", initial_sidebar_state="collapsed")

ACCENT = "#3DD9B0"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root { --accent:#3DD9B0; --bg:#0B0F17; --card:#111726; --card2:#161C2A;
        --line:#1C2536; --text:#F4F6FB; --muted:#8A94A6; --dim:#5F6B7E;
        --buy-bg:#0F2A20; --buy:#4FD1A0; --short-bg:#2A1315; --short:#F0837F; }
.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { padding-top:2rem; max-width:1100px; }

h1 { font-size:1.5rem !important; font-weight:600 !important; color:var(--text) !important;
     letter-spacing:0.01em; display:flex; align-items:center; gap:10px; }
h1::before { content:"S"; display:inline-flex; align-items:center; justify-content:center;
     width:30px; height:30px; border-radius:8px; background:var(--accent);
     color:var(--bg); font-size:17px; font-weight:600; }

.sub { color:var(--dim); font-size:13px; margin:-6px 0 20px 40px; }

.cardrow { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:24px; }
.metric { border-radius:12px; padding:16px 18px; }
.metric .lbl { font-size:12px; margin-bottom:6px; }
.metric .val { color:var(--text); font-size:28px; font-weight:600; line-height:1; }
.m-buy { background:var(--buy-bg); } .m-buy .lbl { color:var(--buy); }
.m-short { background:var(--short-bg); } .m-short .lbl { color:var(--short); }
.m-hold { background:var(--card2); } .m-hold .lbl { color:var(--muted); }

.section { color:var(--muted); font-size:12px; font-weight:600; letter-spacing:0.06em;
     margin:4px 0 12px; text-transform:uppercase; }

.setup { background:var(--card); border-radius:12px; padding:14px 16px; margin-bottom:8px;
     display:flex; align-items:center; justify-content:space-between; }
.setup .left { display:flex; align-items:center; gap:12px; }
.pill { font-size:11px; font-weight:600; padding:4px 9px; border-radius:6px; }
.pill.buy { background:var(--buy-bg); color:var(--buy); }
.pill.short { background:var(--short-bg); color:var(--short); }
.setup .tkr { color:var(--text); font-size:15px; font-weight:600; line-height:1.2; }
.setup .meta { color:var(--dim); font-size:11px; }
.setup .price { color:var(--text); font-size:15px; text-align:right; line-height:1.2; }
.setup .chg { font-size:12px; text-align:right; }
.up { color:var(--buy); } .down { color:var(--short); }

button[data-baseweb="tab"] { font-family:'Inter',sans-serif; color:var(--muted); font-size:13px; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
div[data-baseweb="tab-highlight"] { background-color:var(--accent); }
[data-testid="stSidebar"] { background:var(--bg); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { font-family:'Inter',sans-serif; }
.stCaption, small { color:var(--dim) !important; }
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
    return (f"{days}+" if capped else str(days)), f"{since:%m-%d}", entry, chg


def backtest(sig, close):
    pos = sig.map({"BUY": 1, "SHORT": -1, "HOLD": 0}).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strat = float((1 + pos * ret).prod() - 1)
    bh = float(close.iloc[-1] / close.iloc[0] - 1)
    trades, cur, entry_i = [], 0, None
    for i, p in enumerate(pos.tolist()):
        if p != cur:
            if cur != 0 and entry_i is not None:
                trades.append((float(close.iloc[i - 1]) / float(close.iloc[entry_i]) - 1) * cur)
            cur, entry_i = p, (i if p != 0 else None)
    if cur != 0 and entry_i is not None:
        trades.append((float(close.iloc[-1]) / float(close.iloc[entry_i]) - 1) * cur)
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) * 100 if trades else 0.0
    return strat * 100, bh * 100, len(trades), wr


@st.cache_data(ttl=900, show_spinner=False)
def load(tickers, period, bucket):
    return yf.download(tickers, period=period, interval="1d",
                       progress=False, auto_adjust=True, group_by="ticker")


st.title("Stock Scanner")
if not tickers:
    st.info("Add a ticker in the sidebar."); st.stop()

bucket = int(datetime.now().timestamp() // 60) if live else 0
with st.spinner(f"Scanning {len(tickers)} tickers..."):
    raw = load(tickers, period, bucket)
if raw is None:
    st.error("No data returned."); st.stop()

rows, bt_rows, failed = [], [], []
for t in tickers:
    try:
        cf = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
        if len(cf) < 60: failed.append(t); continue
        ss, cv, r = daily_signals(cf)
        if ss.empty: failed.append(t); continue
        sgn = ss.iloc[-1]
        days, since, entry, chg_since = signal_age(ss, cv)
        rows.append({"ticker": t, "signal": sgn, "days": days, "since": since,
                     "entry": entry, "chg_since_%": chg_since,
                     "close": float(cv.iloc[-1]),
                     "chg_1d_%": (float(cv.iloc[-1]) / float(cv.iloc[-2]) - 1) * 100,
                     "rsi": r["rsi"], "reasons": reasons_text(r, float(cv.iloc[-1]))})
        strat, bh, ntr, wr = backtest(ss, cv)
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

upd = f" &middot; live {datetime.now():%H:%M:%S}" if live else ""
st.markdown(f'<div class="sub">{len(df)} scanned &middot; {date.today():%a %d %b %Y}{upd}</div>',
            unsafe_allow_html=True)

st.markdown(
    f'<div class="cardrow">'
    f'<div class="metric m-buy"><div class="lbl">Buy setups</div><div class="val">{b}</div></div>'
    f'<div class="metric m-short"><div class="lbl">Short setups</div><div class="val">{s}</div></div>'
    f'<div class="metric m-hold"><div class="lbl">Holding</div><div class="val">{h}</div></div>'
    f'</div>', unsafe_allow_html=True)


def setup_card(r):
    cls = "buy" if r["signal"] == "BUY" else "short"
    ch = r["chg_1d_%"]
    updown = "up" if ch >= 0 else "down"
    sign = "+" if ch >= 0 else ""
    return (f'<div class="setup"><div class="left">'
            f'<span class="pill {cls}">{r["signal"]}</span>'
            f'<div><div class="tkr">{r["ticker"]}</div>'
            f'<div class="meta">{r["days"]} sessions &middot; since {r["since"]}</div></div></div>'
            f'<div><div class="price">{r["close"]:.2f}</div>'
            f'<div class="chg {updown}">{sign}{ch:.2f}%</div></div></div>')


def cs(v):
    if v == "BUY": return "color:#4FD1A0;font-weight:600"
    if v == "SHORT": return "color:#F0837F;font-weight:600"
    return "color:#8A94A6"


def cc(v):
    try: return "color:#4FD1A0" if v >= 0 else "color:#F0837F"
    except TypeError: return ""


tab_setups, tab_all, tab_bt = st.tabs(["Setups", "All tickers", "Backtest"])

with tab_setups:
    setups = df[df["signal"] != "HOLD"].copy()
    order = {"BUY": 0, "SHORT": 1}
    setups = setups.sort_values("signal", key=lambda c: c.map(order))
    if setups.empty:
        st.info("No buy or short setups right now. Everything is holding.")
    else:
        st.markdown('<div class="section">Setups today</div>', unsafe_allow_html=True)
        st.markdown("".join(setup_card(r) for _, r in setups.iterrows()),
                    unsafe_allow_html=True)

with tab_all:
    cols = ["ticker", "signal", "days", "since", "entry", "chg_since_%",
            "close", "chg_1d_%", "rsi", "reasons"]
    st.dataframe(
        df[cols].style.map(cs, subset=["signal"])
        .map(cc, subset=["chg_since_%", "chg_1d_%"]).format(precision=2),
        use_container_width=True, height=560)

with tab_bt:
    bts = bt.sort_values("edge_%", ascending=False)
    st.dataframe(
        bts.style.map(cc, subset=["strategy_%", "buy_hold_%", "edge_%"]).format(precision=1),
        use_container_width=True, height=520)
    beat = int((bt["edge_%"] > 0).sum())
    st.caption(f"Following every signal over {period} beat buy-and-hold on {beat} of "
               f"{len(bt)} tickers (avg edge {bt['edge_%'].mean():+.1f}%). "
               "Simulated on daily closes, no fees. Past results do not predict future ones.")

st.divider()
pick = st.selectbox("Chart", df["ticker"])
cc2 = raw[pick]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
st.line_chart(pd.DataFrame({"Close": cc2, "SMA 20": cc2.rolling(20).mean(),
                            "SMA 50": cc2.rolling(50).mean()}),
              height=360, color=["#3DD9B0", "#5B6B85", "#33415C"])
row = df[df["ticker"] == pick].iloc[0]
st.markdown(f"`{pick}`  {row['signal']} since {row['since']}  |  {row['days']} sessions  |  "
            f"{row['chg_since_%']:+.1f}% since signal  |  {row['reasons']}")
st.caption("Simple indicator rules (RSI, MACD, SMA, Bollinger). Screening ideas, not financial advice.")
