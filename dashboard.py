"""
Cam's Stock Scanner — dashboard.py

Signal engine rebuilt to fix three failure modes in the original:
  1. Score arithmetic left no real HOLD band (neutral score could only be +2/0/-2
     against a +/-2 threshold), so tickers flipped BUY<->SHORT on MACD+SMA alone.
  2. Nothing in the score knew where price actually was, so a ticker could sit
     SHORT while trading above its own 20 SMA and making new highs.
  3. Futures were scanned on daily bars, which can never sync with an intraday
     scalping timeframe.

Run:  streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, datetime

try:
    import options_scanner
    OPTIONS_OK = True
except ImportError:
    OPTIONS_OK = False

st.set_page_config(page_title="Stock Scanner", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- style

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root { --accent:#3DD9B0; --bg:#0B0F17; --card:#111726; --card2:#161C2A;
        --line:#1C2536; --text:#F4F6FB; --muted:#8A94A6; --dim:#5F6B7E;
        --buy-bg:#0F2A20; --buy:#4FD1A0; --short-bg:#2A1315; --short:#F0837F;
        --warn-bg:#2A2413; --warn:#E0B84F; }
.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu, footer { visibility:hidden; height:0; }
/* Deliberately NOT hiding stHeader or stToolbar: depending on the Streamlit
   version, the sidebar open/close control lives inside one of them, and
   hiding it leaves no way to reach the sidebar at all. */
header[data-testid="stHeader"] { background:transparent; }
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
.pill { font-size:11px; font-weight:600; padding:4px 9px; border-radius:6px;
     min-width:52px; text-align:center; display:inline-block; }
.pill.buy { background:var(--buy-bg); color:var(--buy); }
.pill.short { background:var(--short-bg); color:var(--short); }
.setup .tkr { color:var(--text); font-size:15px; font-weight:600; line-height:1.2; }
.setup .meta { color:var(--dim); font-size:11px; }
.setup .price { color:var(--text); font-size:15px; text-align:right; line-height:1.2; }
.setup .chg { font-size:12px; text-align:right; }
.up { color:var(--buy); } .down { color:var(--short); }
.tag { font-size:10px; font-weight:600; padding:2px 6px; border-radius:4px;
     background:var(--warn-bg); color:var(--warn); margin-left:6px; }
.tf { font-size:10px; color:var(--dim); border:1px solid var(--line);
     padding:2px 6px; border-radius:4px; margin-left:6px; }

.note { background:var(--card2); border-left:2px solid var(--warn); color:var(--muted);
     font-size:12px; padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:16px; }

.opt { background:var(--card); border-radius:12px; padding:14px 16px; margin-bottom:8px; }
.opt .top { display:flex; align-items:center; justify-content:space-between;
     margin-bottom:10px; }
.opt .contract { color:var(--text); font-size:15px; font-weight:600; }
.opt .cost { color:var(--text); font-size:15px; font-weight:600; text-align:right; }
.opt .cost small { display:block; color:var(--dim); font-size:11px; font-weight:400; }
.opt .grid5 { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }
.opt .cell { background:var(--card2); border-radius:8px; padding:8px 10px; }
.opt .cell .k { color:var(--dim); font-size:10px; letter-spacing:0.05em;
     text-transform:uppercase; margin-bottom:3px; }
.opt .cell .v { color:var(--text); font-size:14px; font-weight:600; }
.opt .v.good { color:var(--buy); } .opt .v.bad { color:var(--short); }
.opt .v.mid { color:var(--warn); }

button[data-baseweb="tab"] { font-family:'Inter',sans-serif; color:var(--muted); font-size:13px; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
div[data-baseweb="tab-highlight"] { background-color:var(--accent); }
[data-testid="stSidebar"] { background:var(--bg); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { font-family:'Inter',sans-serif; }
.stCaption, small { color:var(--dim) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- config

DEFAULT = ("AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,"
           "SPY,QQQ,PLTR,COIN,SOFI,DIS,BA,UBER,SHOP,INTC,"
           "ES=F,MES=F,NQ=F,MNQ=F,YM=F,RTY=F")

st.sidebar.header("Watchlist")
tickers = [t.strip().upper() for t in
           st.sidebar.text_area("Tickers (comma separated)", DEFAULT, height=140).split(",")
           if t.strip()]

st.sidebar.header("Timeframes")
eq_period = st.sidebar.selectbox("Equity history", ["6mo", "1y", "2y"], index=1)
fut_interval = st.sidebar.selectbox("Futures interval", ["5m", "15m", "30m", "1h"], index=1)
scan_futures_intraday = st.sidebar.toggle("Scan futures intraday", value=True,
                                          help="Off = treat futures like equities on daily bars.")

st.sidebar.header("Signal rules")
rsi_buy = st.sidebar.slider("RSI oversold (buy)", 10, 40, 30)
rsi_short = st.sidebar.slider("RSI overbought (short)", 60, 90, 70)
threshold = st.sidebar.slider("Score threshold", 2, 5, 3,
                              help="Higher = fewer, higher-conviction signals. "
                                   "Score runs -6 to +6.")
max_age = st.sidebar.slider("Expire signal after (bars)", 3, 30, 10)
use_veto = st.sidebar.toggle("SMA20 veto", value=True,
                             help="Never short above the 20 SMA, never buy below it.")
live = st.sidebar.toggle("Live refresh (60s)", value=False)

FAST, SLOW, RSI_N, BB_N = 20, 50, 14, 20
MIN_BARS = SLOW + 10
ROLL_GAP = 0.02          # single-bar move that suggests a contract roll splice
STALE_H = {"1d": 48, "1h": 4, "30m": 2, "15m": 2, "5m": 1}

# ---------------------------------------------------------------- data


@st.cache_data(ttl=900, show_spinner=False)
def load(symbols, period, interval, _bucket):
    if not symbols:
        return None
    return yf.download(list(symbols), period=period, interval=interval,
                       progress=False, auto_adjust=True, group_by="ticker")


def extract_close(raw, ticker):
    """Pull one Close series out of whatever shape yfinance handed back."""
    if raw is None or getattr(raw, "empty", True):
        return None
    try:
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            if ticker in cols.get_level_values(0):
                s = raw[ticker]["Close"]
            elif ticker in cols.get_level_values(1):
                s = raw["Close"][ticker]
            else:
                return None
        else:
            s = raw["Close"]
    except (KeyError, IndexError):
        return None
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    return s if len(s) else None


def bar_age_hours(ts):
    tz = getattr(ts, "tz", None) or getattr(ts, "tzinfo", None)
    now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
    return (now - ts).total_seconds() / 3600.0


def roll_gap_bars(close, lookback=60):
    """Index positions where a single bar moved enough to look like a roll splice."""
    tail = close.tail(lookback)
    jumps = tail.pct_change().abs()
    return tail.index[jumps > ROLL_GAP]


# ---------------------------------------------------------------- signals


def indicators(close):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1 / RSI_N, adjust=False, min_periods=RSI_N).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_N, adjust=False, min_periods=RSI_N).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    sma_f = close.rolling(FAST).mean()
    sma_s = close.rolling(SLOW).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    std = close.rolling(BB_N).std()
    bb_up = close.rolling(BB_N).mean() + 2 * std
    bb_lo = close.rolling(BB_N).mean() - 2 * std
    return dict(rsi=rsi, sma_f=sma_f, sma_s=sma_s, macd=macd,
                macd_sig=macd_sig, bb_up=bb_up, bb_lo=bb_lo)


def score_series(close, ind):
    """Range -6..+6. Every term is signed, so a genuine HOLD band exists."""
    def pm(cond):                      # +1 / -1
        return cond.astype(int) * 2 - 1

    return (
        (ind["rsi"] < rsi_buy).astype(int) * 2          # oversold
        - (ind["rsi"] > rsi_short).astype(int) * 2      # overbought
        + pm(ind["macd"] > ind["macd_sig"])             # momentum
        + pm(ind["sma_f"] > ind["sma_s"])               # trend
        + pm(close > ind["sma_f"])                      # where price actually is
        + (close <= ind["bb_lo"]).astype(int)
        - (close >= ind["bb_up"]).astype(int)
    )


def build_signals(close):
    ind = indicators(close)
    score = score_series(close, ind)

    sig = pd.Series("HOLD", index=close.index)
    sig[score >= threshold] = "BUY"
    sig[score <= -threshold] = "SHORT"

    vetoed = pd.Series(False, index=close.index)
    if use_veto:
        bad_short = (sig == "SHORT") & (close > ind["sma_f"])
        bad_buy = (sig == "BUY") & (close < ind["sma_f"])
        vetoed = bad_short | bad_buy
        sig[vetoed] = "HOLD"

    # expire anything that has run past max_age consecutive bars
    run = sig.groupby((sig != sig.shift()).cumsum()).cumcount()
    expired = (run >= max_age) & (sig != "HOLD")
    sig[expired] = "HOLD"

    valid = ind["sma_s"].notna()
    snap = {k: v.iloc[-1] for k, v in ind.items()}
    snap["score"] = float(score.iloc[-1])
    snap["vetoed"] = bool(vetoed.iloc[-1])
    snap["expired"] = bool(expired.iloc[-1])
    return sig[valid], close[valid], snap


def reasons_text(snap, last):
    why = []
    r = snap["rsi"]
    if r < rsi_buy:
        why.append(f"RSI {r:.0f} oversold")
    elif r > rsi_short:
        why.append(f"RSI {r:.0f} overbought")
    else:
        why.append(f"RSI {r:.0f}")
    why.append("MACD bullish" if snap["macd"] > snap["macd_sig"] else "MACD bearish")
    why.append("Uptrend" if snap["sma_f"] > snap["sma_s"] else "Downtrend")
    why.append("Above 20 SMA" if last > snap["sma_f"] else "Below 20 SMA")
    if last <= snap["bb_lo"]:
        why.append("Lower Bollinger")
    elif last >= snap["bb_up"]:
        why.append("Upper Bollinger")
    if snap["vetoed"]:
        why.append("VETOED by SMA20 rule")
    if snap["expired"]:
        why.append(f"EXPIRED past {max_age} bars")
    return " | ".join(why)


def signal_age(sig, close):
    current = sig.iloc[-1]
    flip = 0
    for i in range(len(sig) - 1, -1, -1):
        if sig.iloc[i] != current:
            flip = i + 1
            break
    bars = len(sig) - flip
    capped = flip == 0
    entry = float(close.iloc[flip])
    chg = (float(close.iloc[-1]) / entry - 1) * 100
    return (f"{bars}+" if capped else str(bars)), sig.index[flip], entry, chg


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


# ---------------------------------------------------------------- scan

st.title("Stock Scanner")
if not tickers:
    st.info("Add a ticker in the sidebar.")
    st.stop()

futures = [t for t in tickers if t.endswith("=F")]
equities = [t for t in tickers if not t.endswith("=F")]
if not scan_futures_intraday:
    equities, futures = tickers, []

bucket = int(datetime.now().timestamp() // 60) if live else 0
with st.spinner(f"Scanning {len(tickers)} tickers..."):
    raw_eq = load(tuple(equities), eq_period, "1d", bucket)
    raw_fut = load(tuple(futures), "60d", fut_interval, bucket) if futures else None

sources = {t: (raw_eq, "1d") for t in equities}
sources.update({t: (raw_fut, fut_interval) for t in futures})

rows, bt_rows, failed = [], [], []
for t in tickers:
    raw, interval = sources[t]
    close = extract_close(raw, t)
    if close is None or len(close) < MIN_BARS:
        failed.append(t)
        continue
    try:
        sig, cv, snap = build_signals(close)
        if sig.empty:
            failed.append(t)
            continue

        bars, since_ts, entry, chg_since = signal_age(sig, cv)
        last = float(cv.iloc[-1])
        prev = float(cv.iloc[-2])
        age_h = bar_age_hours(cv.index[-1])
        stale = age_h > STALE_H.get(interval, 48)
        gaps = roll_gap_bars(cv) if t.endswith("=F") else []

        rows.append({
            "ticker": t, "signal": sig.iloc[-1], "tf": interval,
            "bars": bars, "since": f"{since_ts:%m-%d %H:%M}" if interval != "1d"
                                    else f"{since_ts:%m-%d}",
            "entry": entry, "chg_since_%": chg_since, "close": last,
            "chg_1bar_%": (last / prev - 1) * 100,
            "score": snap["score"], "rsi": snap["rsi"], "sma20": snap["sma_f"],
            "stale": stale, "bar_age_h": age_h, "roll_gap": len(gaps) > 0,
            "reasons": reasons_text(snap, last),
        })

        strat, bh, ntr, wr = backtest(sig, cv)
        bt_rows.append({"ticker": t, "tf": interval, "strategy_%": strat,
                        "buy_hold_%": bh, "edge_%": strat - bh,
                        "trades": ntr, "win_rate_%": wr})
    except Exception:
        failed.append(t)

if failed:
    st.sidebar.warning("No data for: " + ", ".join(failed))
if not rows:
    st.error("No data returned. Check the ticker list, then reload.")
    st.stop()

df = pd.DataFrame(rows)
bt = pd.DataFrame(bt_rows)
n_buy = int((df["signal"] == "BUY").sum())
n_short = int((df["signal"] == "SHORT").sum())
n_hold = len(df) - n_buy - n_short

upd = f" &middot; live {datetime.now():%H:%M:%S}" if live else ""
st.markdown(f'<div class="sub">{len(df)} scanned &middot; {date.today():%a %d %b %Y}{upd}</div>',
            unsafe_allow_html=True)

st.markdown(
    f'<div class="cardrow">'
    f'<div class="metric m-buy"><div class="lbl">Buy setups</div><div class="val">{n_buy}</div></div>'
    f'<div class="metric m-short"><div class="lbl">Short setups</div><div class="val">{n_short}</div></div>'
    f'<div class="metric m-hold"><div class="lbl">Holding</div><div class="val">{n_hold}</div></div>'
    f'</div>', unsafe_allow_html=True)

stale_list = df.loc[df["stale"], "ticker"].tolist()
if stale_list:
    st.markdown(
        f'<div class="note">Last bar is older than the current session for '
        f'<b>{len(stale_list)}</b> ticker(s): {", ".join(stale_list[:12])}'
        f'{"..." if len(stale_list) > 12 else ""}. '
        f'Signals here are computed on closed bars and will not reflect anything '
        f'trading right now.</div>', unsafe_allow_html=True)

roll_list = df.loc[df["roll_gap"], "ticker"].tolist()
if roll_list:
    st.markdown(
        f'<div class="note">Possible contract roll splice in the recent history for '
        f'{", ".join(roll_list)}. Continuous futures series stitch contracts together, '
        f'and the price jump at the seam feeds MACD and Bollinger as if it were a real '
        f'move. Treat signals near the seam with suspicion.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- render


def setup_card(r):
    cls = "buy" if r["signal"] == "BUY" else "short"
    ch = r["chg_1bar_%"]
    ud, sign = ("up", "+") if ch >= 0 else ("down", "")
    tags = ""
    if r["stale"]:
        tags += '<span class="tag">STALE</span>'
    if r["roll_gap"]:
        tags += '<span class="tag">ROLL</span>'
    tf = f'<span class="tf">{r["tf"]}</span>'
    return (f'<div class="setup"><div class="left">'
            f'<span class="pill {cls}">{r["signal"]}</span>'
            f'<div><div class="tkr">{r["ticker"]}{tf}{tags}</div>'
            f'<div class="meta">score {r["score"]:+.0f} &middot; {r["bars"]} bars '
            f'&middot; since {r["since"]}</div></div></div>'
            f'<div><div class="price">{r["close"]:.2f}</div>'
            f'<div class="chg {ud}">{sign}{ch:.2f}%</div></div></div>')


def color_sig(v):
    return {"BUY": "color:#4FD1A0;font-weight:600",
            "SHORT": "color:#F0837F;font-weight:600"}.get(v, "color:#8A94A6")


def color_num(v):
    try:
        return "color:#4FD1A0" if v >= 0 else "color:#F0837F"
    except TypeError:
        return ""


def prob_class(p):
    return "bad" if p < 20 else ("mid" if p < 40 else "good")


def option_card(o):
    sd = "buy" if o["side"] == "BUY" else "short"
    label = f'{o["ticker"]} ${o["strike"]:g} {o["kind"].upper()} {o["expiration"]}'
    mv_cls = "bad" if abs(o["req_move"]) > 10 else ("mid" if abs(o["req_move"]) > 5 else "good")
    sp_cls = "bad" if o["spread_pct"] > 15 else ("mid" if o["spread_pct"] > 8 else "good")
    return (
        f'<div class="opt"><div class="top">'
        f'<div><span class="pill {sd}">{o["side"]}</span> '
        f'<span class="contract">{label}</span></div>'
        f'<div class="cost">${o["cost"]:,.0f}<small>per contract</small></div></div>'
        f'<div class="grid5">'
        f'<div class="cell"><div class="k">Prob ITM</div>'
        f'<div class="v {prob_class(o["prob_itm"])}">{o["prob_itm"]:.0f}%</div></div>'
        f'<div class="cell"><div class="k">Move needed</div>'
        f'<div class="v {mv_cls}">{o["req_move"]:+.1f}%</div></div>'
        f'<div class="cell"><div class="k">Breakeven</div>'
        f'<div class="v">${o["breakeven"]:,.2f}</div></div>'
        f'<div class="cell"><div class="k">Decay/day</div>'
        f'<div class="v">{abs(o["theta_pct"]):.1f}%</div></div>'
        f'<div class="cell"><div class="k">Spread</div>'
        f'<div class="v {sp_cls}">{o["spread_pct"]:.0f}%</div></div>'
        f'</div></div>')


tab_setups, tab_all, tab_opt, tab_bt = st.tabs(
    ["Setups", "All tickers", "Options", "Backtest"])

with tab_setups:
    setups = df[df["signal"] != "HOLD"].copy()
    setups = setups.sort_values(
        ["signal", "score"], key=lambda c: c.map({"BUY": 0, "SHORT": 1}) if c.name == "signal" else c)
    if setups.empty:
        st.info("No buy or short setups right now. Everything is holding.")
    else:
        st.markdown('<div class="section">Setups</div>', unsafe_allow_html=True)
        st.markdown("".join(setup_card(r) for _, r in setups.iterrows()),
                    unsafe_allow_html=True)

with tab_all:
    cols = ["ticker", "signal", "tf", "score", "bars", "since", "entry",
            "chg_since_%", "close", "chg_1bar_%", "rsi", "stale", "reasons"]
    st.dataframe(
        df[cols].style.map(color_sig, subset=["signal"])
                      .map(color_num, subset=["chg_since_%", "chg_1bar_%", "score"])
                      .format(precision=2),
        use_container_width=True, height=560)
    st.caption("The reasons column shows every term that fed the score, including "
               "any signal that was vetoed or expired. If a ticker looks wrong, read "
               "that row first.")

with tab_opt:
    if not OPTIONS_OK:
        st.error("`options_scanner.py` not found. Drop it in this folder and reload.")
    else:
        # These live in the tab, not only the sidebar, so the feature is
        # reachable on mobile and on any Streamlit version regardless of
        # whether the sidebar control is rendered.
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            opt_enable = st.toggle(
                "Scan options on setups", value=False, key="opt_go",
                help="Pulls option chains for tickers currently showing BUY or "
                     "SHORT. Slow — one request per expiration per ticker.")
        with c2:
            opt_target_delta = st.slider("Target delta", 0.15, 0.70, 0.40, 0.05,
                                         key="opt_delta")
        with c3:
            opt_min_oi = st.number_input("Min open interest", 0, 5000, 100, 50,
                                         key="opt_oi")
        d1, d2 = st.columns(2)
        with d1:
            opt_min_dte = st.slider("Min days to expiration", 7, 60, 21, key="opt_lo")
        with d2:
            opt_max_dte = st.slider("Max days to expiration", 21, 120, 60, key="opt_hi")

        if opt_min_dte >= opt_max_dte:
            st.warning("Min days must be below max days.")
            st.stop()

        if not opt_enable:
            st.info("Flip the toggle above to price the current setups. "
                    "It only pulls chains for tickers already showing BUY or "
                    "SHORT, so it stays cheap.")
        else:
            live_setups = df[df["signal"] != "HOLD"]
            live_setups = live_setups[~live_setups["ticker"].str.endswith("=F")]
            if live_setups.empty:
                st.info("No equity setups right now, so there is nothing to price. "
                        "Futures options are not covered here.")
            else:
                smap = dict(zip(live_setups["ticker"], live_setups["signal"]))
                with st.spinner(f"Pulling chains for {len(smap)} ticker(s)..."):
                    try:
                        opts = options_scanner.best_contracts(
                            list(smap.keys()), signal_map=smap,
                            target_delta=opt_target_delta,
                            min_dte=opt_min_dte, max_dte=opt_max_dte,
                            min_open_interest=int(opt_min_oi))
                    except Exception as e:
                        opts = []
                        st.error(f"Options fetch failed: {e}")

                if not opts:
                    st.warning("Nothing cleared the liquidity filters. That is a normal "
                               "result — most contracts on most names are not worth "
                               "trading. Loosen open interest or widen the DTE window "
                               "if you want to see what got cut.")
                else:
                    st.markdown('<div class="section">Cheapest sane way to express each setup</div>',
                                unsafe_allow_html=True)
                    st.markdown("".join(option_card(o) for o in opts),
                                unsafe_allow_html=True)

                    worst = min(opts, key=lambda o: o["prob_itm"])
                    st.caption(
                        f"Probability ITM is the risk-neutral N(d2) from Black-Scholes on "
                        f"the chain's own implied vol — it is the market's estimate, not "
                        f"mine. Lowest on screen is {worst['ticker']} at "
                        f"{worst['prob_itm']:.0f}%. Anything under 20% loses its full "
                        f"premium most of the time, and the decay column is how fast that "
                        f"happens while you wait. Contracts are 100 shares, so the cost "
                        f"shown is real money per contract.")

with tab_bt:
    bts = bt.sort_values("edge_%", ascending=False)
    st.dataframe(
        bts.style.map(color_num, subset=["strategy_%", "buy_hold_%", "edge_%"])
                 .format(precision=1),
        use_container_width=True, height=520)
    beat = int((bt["edge_%"] > 0).sum())
    st.caption(f"Following every signal beat buy-and-hold on {beat} of {len(bt)} tickers "
               f"(avg edge {bt['edge_%'].mean():+.1f}%). Simulated on closes with no fees, "
               f"no slippage and no overnight gaps modelled, and intraday rows cover only "
               f"the last 60 days. Past results do not predict future ones.")

st.divider()
pick = st.selectbox("Chart", df["ticker"])
raw_pick, interval_pick = sources[pick]
series = extract_close(raw_pick, pick)
st.line_chart(
    pd.DataFrame({"Close": series,
                  f"SMA {FAST}": series.rolling(FAST).mean(),
                  f"SMA {SLOW}": series.rolling(SLOW).mean()}),
    height=360, color=["#3DD9B0", "#5B6B85", "#33415C"])

row = df[df["ticker"] == pick].iloc[0]
st.markdown(f"`{pick}` &nbsp;{row['signal']} &nbsp;|&nbsp; {interval_pick} bars "
            f"&nbsp;|&nbsp; score {row['score']:+.0f} &nbsp;|&nbsp; {row['bars']} bars "
            f"since {row['since']} &nbsp;|&nbsp; {row['chg_since_%']:+.1f}% since signal")
st.markdown(f"<div class='meta' style='color:#5F6B7E;font-size:12px'>{row['reasons']}</div>",
            unsafe_allow_html=True)
st.caption("Screening ideas from simple indicator rules, not financial advice. "
           "I'm not a licensed advisor — size and risk decisions are yours.")
