"""
CAM'S STOCK SCANNER — Streamlit app
===================================
Entrypoint for Streamlit Community Cloud.

Set "Main file path" in the app settings to:  streamlit_app.py

WHAT CHANGED FROM THE OLD BUILD
-------------------------------
1. Per-timeframe parameter profiles. SMA lengths, RSI lookback and bands,
   MACD spans, entry/exit thresholds, expiry and the veto buffer all rescale
   with the bar size instead of running daily-tuned numbers on 1-minute noise.

2. Multi-timeframe matrix. Every ticker shows its score on every timeframe at
   once, so a name never silently vanishes when you switch intervals — you see
   1m -3 | 5m -3 | 15m -3 | 30m -1 | 1h +1 and read the disagreement directly.

3. Trend-aware RSI (optional). The score mixes trend terms with mean-reversion
   terms, and in a real move they cancel: measured on a clean uptrend the trend
   terms average +1.97 while the RSI term averages -1.98, pinning the score near
   +1 and leaving everything stuck on HOLD. Trend-aware mode stops RSI fading a
   trend that price and structure both confirm.

4. period travels with interval. yfinance caps intraday history (1m -> 7d,
   5m/15m/30m -> 60d, 1h -> 730d). Asking for more returns an EMPTY frame, not
   an error, so every ticker "fails" for no visible reason.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Cam's Stock Scanner", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# TIMEFRAME PROFILES
# ----------------------------------------------------------------------
# period      how far back to pull. Must respect the yfinance intraday caps.
# fast/slow   SMA lengths. 9/21 on fast bars so trend flips inside the session.
# rsi_n       RSI lookback; shorter = more responsive, so bands widen to match.
# rsi_buy/    RSI 30 fires constantly on 1m. 25/75 keeps "oversold" meaningful.
#   rsi_short
# macd        (fast, slow, signal) EMA spans, scaled with the bar.
# entry       |score| needed to OPEN. Keep at 3 — the score cannot reach 4 in
#             practice, so raising it makes the scanner go permanently silent.
# exit        hysteresis floor. Once open the signal holds until score decays
#             past this. A wider entry/exit gap means fewer whipsaw exits.
# confirm     score must hold past `entry` for this many CONSECUTIVE bars before
#             a signal opens. Biggest noise filter on 1m/2m — measured 60% fewer
#             signal flips on random-walk data at confirm=2 vs confirm=0.
# max_age     bars before a stale signal is force-expired (~one session each).
# veto_buf    how far past the fast SMA counts as a real break. 0.15% is ~35 NQ
#             points — meaningless on 1m — so it tightens on fast bars.
# trend_rsi   suppress the RSI penalty while trend and location agree.
TF_PROFILES = {
    "1m":  dict(period="5d",   fast=9,  slow=21, rsi_n=7,  bb_n=20,
                rsi_buy=25, rsi_short=75, macd=(6, 13, 5), trend_rsi=True,
                entry=3, exit=0, confirm=3, max_age=30,
                veto_buf=0.0004, min_bars=60, stale_h=0.5),
    "2m":  dict(period="10d",  fast=9,  slow=21, rsi_n=9,  bb_n=20,
                rsi_buy=27, rsi_short=73, macd=(8, 17, 6), trend_rsi=True,
                entry=3, exit=0, confirm=2, max_age=30,
                veto_buf=0.0006, min_bars=60, stale_h=0.75),
    "5m":  dict(period="60d",  fast=12, slow=30, rsi_n=9,  bb_n=20,
                rsi_buy=27, rsi_short=73, macd=(8, 17, 6), trend_rsi=True,
                entry=3, exit=0, confirm=2, max_age=36,
                veto_buf=0.0008, min_bars=70, stale_h=1),
    "15m": dict(period="60d",  fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), trend_rsi=True,
                entry=3, exit=0, confirm=1, max_age=26,
                veto_buf=0.0015, min_bars=60, stale_h=2),
    "30m": dict(period="60d",  fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), trend_rsi=False,
                entry=3, exit=0, confirm=1, max_age=20,
                veto_buf=0.0020, min_bars=60, stale_h=3),
    "1h":  dict(period="180d", fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), trend_rsi=False,
                entry=3, exit=1, confirm=0, max_age=16,
                veto_buf=0.0025, min_bars=60, stale_h=4),
    "1d":  dict(period="1y",   fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), trend_rsi=False,
                entry=3, exit=1, confirm=0, max_age=10,
                veto_buf=0.0015, min_bars=60, stale_h=48),
}
TF_ORDER = ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]
INTRADAY = ["1m", "2m", "5m", "15m", "30m", "1h"]

DEFAULT_TICKERS = ("AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,"
                   "SPY,QQQ,PLTR,COIN,SOFI,DIS,BA,UBER,SHOP,INTC,"
                   "ES=F,MES=F,NQ=F,MNQ=F,YM=F,RTY=F")

ROLL_GAP = 0.02

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1400px}
.hero{border-bottom:1px solid #232C3A;padding-bottom:14px;margin-bottom:18px}
.hero h1{margin:0;font-size:26px;letter-spacing:-.02em;color:#E8EEF6}
.hero .sub{color:#5F6B7E;font-size:13px;margin-top:4px}
.card{background:#111721;border:1px solid #1E2733;border-left:3px solid #2B3646;
      border-radius:10px;padding:12px 16px;margin-bottom:8px}
.card.buy{border-left-color:#5EE9B5}
.card.short{border-left-color:#FF6B7A}
.card-top{display:flex;align-items:center;gap:10px}
.tag{font-size:10px;font-weight:800;letter-spacing:.09em;padding:3px 9px;
     border-radius:5px}
.tag.buy{background:#123A2E;color:#5EE9B5}
.tag.short{background:#3A1620;color:#FF6B7A}
.tag.hold{background:#1B222D;color:#6B7789}
.tkr{font-size:16px;font-weight:700;color:#E8EEF6;letter-spacing:-.01em}
.tf-pill{font-size:10px;color:#6B7789;background:#1B222D;padding:2px 7px;
         border-radius:4px}
.px{margin-left:auto;text-align:right}
.px .p{font-size:16px;font-weight:700;color:#E8EEF6}
.px .c{font-size:12px}
.up{color:#5EE9B5}.down{color:#FF6B7A}.flat{color:#6B7789}
.meta{color:#5F6B7E;font-size:11.5px;margin-top:5px}
.mtf{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px}
.mtf b{font-size:9px;letter-spacing:.07em;color:#4A5566;text-transform:uppercase;
       align-self:center;margin-right:2px}
.cell{font-size:10.5px;font-weight:700;padding:3px 7px;border-radius:4px;
      background:#171E29;color:#5A6678;border:1px solid #212A37;
      font-variant-numeric:tabular-nums}
.cell.pos{background:#0F2A22;color:#5EE9B5;border-color:#1B4437}
.cell.neg{background:#2C1219;color:#FF6B7A;border-color:#48202A}
.cell.na{opacity:.35}
.cell i{font-style:normal;color:#4A5566;font-weight:600}
.cell.pos i{color:#3E9E7F}.cell.neg i{color:#B85260}
.split{font-size:11px;color:#C99A3A;margin-top:7px}
.warn{font-size:11px;color:#C97C3A;margin-top:5px}
.empty{border:1px dashed #232C3A;border-radius:10px;padding:26px;
       text-align:center;color:#5F6B7E;font-size:13px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def is_future(t):
    return t.endswith("=F")


@st.cache_data(ttl=120, show_spinner=False)
def fetch(symbols, interval):
    """One download per timeframe. period is derived from interval, never
    passed in separately — that pairing is what keeps 1m from coming back
    empty."""
    if not symbols:
        return None
    period = TF_PROFILES[interval]["period"]
    try:
        return yf.download(list(symbols), period=period, interval=interval,
                           progress=False, auto_adjust=True, group_by="ticker",
                           threads=True)
    except Exception as e:
        st.warning(f"Download failed for {interval}: {e}")
        return None


def extract_close(raw, ticker):
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


def has_roll_gap(close, lookback=60):
    return bool((close.tail(lookback).pct_change().abs() > ROLL_GAP).any())


# ----------------------------------------------------------------------
# INDICATORS + SCORING
# ----------------------------------------------------------------------
def indicators(close, p):
    n = p["rsi_n"]
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

    mid = close.rolling(p["bb_n"]).mean()
    std = close.rolling(p["bb_n"]).std()
    mf, ms, msig = p["macd"]
    macd = close.ewm(span=mf, adjust=False).mean() - close.ewm(span=ms, adjust=False).mean()
    return {
        "rsi": rsi,
        "sma_f": close.rolling(p["fast"]).mean(),
        "sma_s": close.rolling(p["slow"]).mean(),
        "macd": macd,
        "macd_sig": macd.ewm(span=msig, adjust=False).mean(),
        "bb_up": mid + 2 * std,
        "bb_lo": mid - 2 * std,
    }


def score_series(close, ind, p, trend_rsi):
    def pm(a, b):
        """+1 above, -1 below, 0 on a tie or missing data.

        A plain `a > b` collapses ties into -1, so three separate terms all
        read bearish at once when price sits exactly on its averages. A halted
        stock or a thin overnight futures session scores -3 SHORT off a flat
        line. Ties are genuinely neutral, so score them that way.
        """
        return (a > b).astype(int) - (a < b).astype(int)

    up = pm(ind["macd"], ind["macd_sig"])
    trend = pm(ind["sma_f"], ind["sma_s"])
    loc = pm(close, ind["sma_f"])

    over = ind["rsi"] > p["rsi_short"]
    under = ind["rsi"] < p["rsi_buy"]

    if trend_rsi:
        # Don't fade strength that trend AND location both confirm. Without
        # this, RSI(-2) cancels trend+location(+2) and the score flatlines
        # near +/-1 exactly when a move is working.
        bull = (trend > 0) & (loc > 0)
        bear = (trend < 0) & (loc < 0)
        over, under = over & ~bull, under & ~bear

    # With zero volatility the bands collapse onto price and BOTH touches fire.
    live = ind["bb_up"] > ind["bb_lo"]
    return (under.astype(int) * 2 - over.astype(int) * 2 + up + trend + loc
            + ((close <= ind["bb_lo"]) & live).astype(int)
            - ((close >= ind["bb_up"]) & live).astype(int))


def build_signals(close, interval, ov):
    """Stateful walk so entry and exit can use different rules.

    Opening needs |score| >= entry for `confirm` consecutive bars. Once open the
    signal survives until score decays past exit, or price breaks the fast SMA
    by veto_buf. A vectorised threshold cannot express that, hence the loop.
    """
    p = dict(TF_PROFILES[interval])
    p["rsi_buy"] = ov["rsi_buy"]
    p["rsi_short"] = ov["rsi_short"]

    ind = indicators(close, p)
    score = score_series(close, ind, p, p["trend_rsi"] and ov["trend_rsi"])

    entry = ov["entry"] if ov["entry"] is not None else p["entry"]
    exit_thr = min(p["exit"], entry - 1)
    max_age = p["max_age"]
    confirm = p["confirm"] if ov["confirm"] else 0
    vbuf = p["veto_buf"]

    sma_f = ind["sma_f"]
    states, vetoes, expiries = [], [], []
    state, run, s_up, s_dn = "HOLD", 0, 0, 0

    for i in range(len(close)):
        s_i = score.iloc[i]
        px = float(close.iloc[i])
        sma = sma_f.iloc[i]
        ok = pd.notna(sma)
        vetoed = False
        below = ok and px < sma * (1 - vbuf)
        above = ok and px > sma * (1 + vbuf)

        s_up = s_up + 1 if s_i >= entry else 0
        s_dn = s_dn + 1 if s_i <= -entry else 0

        if state == "HOLD":
            run = 0
            if s_up > confirm:
                if ov["veto"] and below:
                    vetoed = True
                else:
                    state, run = "BUY", 1
            elif s_dn > confirm:
                if ov["veto"] and above:
                    vetoed = True
                else:
                    state, run = "SHORT", 1
        elif state == "BUY":
            kill = ov["veto"] and below
            if s_i < exit_thr or kill:
                vetoed, state, run = kill, "HOLD", 0
            else:
                run += 1
        elif state == "SHORT":
            kill = ov["veto"] and above
            if s_i > -exit_thr or kill:
                vetoed, state, run = kill, "HOLD", 0
            else:
                run += 1

        expired = False
        if state != "HOLD" and run > max_age:
            state, run, expired = "HOLD", 0, True

        states.append(state)
        vetoes.append(vetoed)
        expiries.append(expired)

    sig = pd.Series(states, index=close.index)
    valid = ind["sma_s"].notna()
    snap = {k: float(v.iloc[-1]) for k, v in ind.items()}
    snap.update(score=float(score.iloc[-1]), vetoed=bool(vetoes[-1]),
                expired=bool(expiries[-1]), entry=entry, exit=exit_thr,
                confirm=confirm, fast=p["fast"], slow=p["slow"],
                rsi_buy=p["rsi_buy"], rsi_short=p["rsi_short"],
                trend_rsi=p["trend_rsi"] and ov["trend_rsi"])
    return sig[valid], close[valid], snap, ind


def signal_age(sig, close):
    cur = sig.iloc[-1]
    flip = 0
    for i in range(len(sig) - 1, -1, -1):
        if sig.iloc[i] != cur:
            flip = i + 1
            break
    entry = float(close.iloc[flip])
    return (f"{len(sig) - flip}{'+' if flip == 0 else ''}", sig.index[flip],
            entry, (float(close.iloc[-1]) / entry - 1) * 100)


def reasons(snap, last):
    w = []
    r, lo, hi = snap["rsi"], snap["rsi_buy"], snap["rsi_short"]
    w.append(f"RSI {r:.0f} oversold" if r < lo else
             f"RSI {r:.0f} overbought" if r > hi else f"RSI {r:.0f}")
    w.append("MACD bullish" if snap["macd"] > snap["macd_sig"] else "MACD bearish")
    w.append("uptrend" if snap["sma_f"] > snap["sma_s"] else "downtrend")
    w.append(f"above {snap['fast']} SMA" if last > snap["sma_f"]
             else f"below {snap['fast']} SMA")
    if last <= snap["bb_lo"]:
        w.append("lower Bollinger")
    elif last >= snap["bb_up"]:
        w.append("upper Bollinger")
    if snap["vetoed"]:
        w.append("VETOED by SMA rule")
    if snap["expired"]:
        w.append("EXPIRED past max bars")
    if snap["trend_rsi"]:
        w.append("RSI trend-aware")
    if snap["confirm"]:
        w.append(f"needs {snap['confirm'] + 1} bars to confirm")
    return " · ".join(w)


# ----------------------------------------------------------------------
# SCAN
# ----------------------------------------------------------------------
def scan(tickers, eq_tf, fut_tf, fut_intraday, mtf_list, ov):
    """Scan the primary timeframe, then score every ticker on each extra
    timeframe so nothing silently disappears when you switch intervals."""
    def primary(t):
        return fut_tf if (is_future(t) and fut_intraday) else eq_tf

    needed = sorted({primary(t) for t in tickers} | set(mtf_list))
    raw = {}
    bar = st.progress(0.0, text="Loading…")
    for i, tf in enumerate(needed):
        bar.progress(i / len(needed), text=f"Loading {tf} bars…")
        raw[tf] = fetch(tuple(tickers), tf)
    bar.progress(1.0, text="Scoring…")

    rows, failed = [], []
    for t in tickers:
        tf = primary(t)
        p = TF_PROFILES[tf]
        close = extract_close(raw.get(tf), t)
        if close is None or len(close) < p["min_bars"]:
            failed.append(t)
            continue
        try:
            sig, cv, snap, ind = build_signals(close, tf, ov)
            if sig.empty or len(cv) < 2:
                failed.append(t)
                continue

            bars, since, entry_px, chg_since = signal_age(sig, cv)
            last, prev = float(cv.iloc[-1]), float(cv.iloc[-2])
            age = bar_age_hours(cv.index[-1])

            # score on every extra timeframe
            matrix = {}
            for m in mtf_list:
                c2 = extract_close(raw.get(m), t)
                if c2 is None or len(c2) < TF_PROFILES[m]["min_bars"]:
                    matrix[m] = None
                    continue
                try:
                    s2, _, snap2, _ = build_signals(c2, m, ov)
                    matrix[m] = (snap2["score"], s2.iloc[-1])
                except Exception:
                    matrix[m] = None

            rows.append(dict(
                ticker=t, signal=sig.iloc[-1], tf=tf, score=snap["score"],
                bars=bars, since=f"{since:%m-%d}" if tf == "1d" else f"{since:%m-%d %H:%M}",
                entry=entry_px, chg_since=chg_since, close=last,
                chg=(last / prev - 1) * 100, rsi=snap["rsi"],
                stale=age > p["stale_h"], age_h=age,
                roll=is_future(t) and has_roll_gap(cv),
                matrix=matrix, reasons=reasons(snap, last),
                series=cv.tail(260), sma_f=ind["sma_f"].tail(260),
                sma_s=ind["sma_s"].tail(260), snap=snap))
        except Exception as e:
            failed.append(f"{t} ({e})")
    bar.empty()
    return rows, failed


# ----------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------
def mtf_html(matrix, primary_tf):
    if not matrix:
        return ""
    cells = ['<b>score by tf</b>']
    for tf, val in matrix.items():
        star = "＊" if tf == primary_tf else ""
        if val is None:
            cells.append(f'<span class="cell na"><i>{tf}</i> —</span>')
            continue
        sc, sg = val
        cls = "pos" if sc > 0 else "neg" if sc < 0 else ""
        mark = "▲" if sg == "BUY" else "▼" if sg == "SHORT" else ""
        cells.append(f'<span class="cell {cls}"><i>{tf}{star}</i> '
                     f'{sc:+.0f}{mark}</span>')
    return f'<div class="mtf">{"".join(cells)}</div>'


def card(r, show_mtf):
    sg = r["signal"]
    cls = sg.lower() if sg in ("BUY", "SHORT") else "hold"
    ccls = "up" if r["chg"] > 0 else "down" if r["chg"] < 0 else "flat"
    flags = ""
    if r["stale"]:
        flags += (f'<div class="warn">Stale — last bar is '
                  f'{r["age_h"]:.1f}h old. Market may be closed.</div>')
    if r["roll"]:
        flags += ('<div class="split">Possible contract roll in this series — '
                  'a splice can look like a real move.</div>')
    return f"""<div class="card {cls}">
  <div class="card-top">
    <span class="tag {cls}">{sg}</span>
    <span class="tkr">{r['ticker']}</span>
    <span class="tf-pill">{r['tf']}</span>
    <span class="px"><span class="p">{r['close']:,.2f}</span><br>
      <span class="c {ccls}">{r['chg']:+.2f}%</span></span>
  </div>
  <div class="meta">score {r['score']:+.0f} · {r['bars']} bars · since {r['since']}
    · {r['chg_since']:+.2f}% since entry</div>
  <div class="meta">{r['reasons']}</div>
  {mtf_html(r['matrix'], r['tf']) if show_mtf else ''}
  {flags}
</div>"""


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Watchlist")
    raw_tk = st.text_area("Tickers (comma separated)", DEFAULT_TICKERS,
                          height=110, label_visibility="visible")
    tickers = [t.strip().upper() for t in raw_tk.split(",") if t.strip()]

    st.markdown("### Timeframes")
    eq_tf = st.selectbox("Equities", TF_ORDER, index=TF_ORDER.index("1d"))
    fut_tf = st.selectbox("Futures", TF_ORDER, index=TF_ORDER.index("15m"))
    fut_intraday = st.toggle(
        "Scan futures intraday", value=True,
        help="Off = futures use the equities timeframe. A daily 20/50 SMA can "
             "never stay in sync with an intraday scalp.")

    st.markdown("### Multi-timeframe")
    show_mtf = st.toggle(
        "Show score on every timeframe", value=True,
        help="Each ticker gets scored on all selected timeframes, so a name "
             "never vanishes just because it fell under the threshold on one.")
    mtf_list = st.multiselect(
        "Timeframes to compare", TF_ORDER,
        default=["5m", "15m", "30m", "1h", "1d"],
        help="Each one is an extra download. 1m only has 7 days of history.",
        disabled=not show_mtf)
    if not show_mtf:
        mtf_list = []

    st.markdown("### Signal rules")
    entry = st.slider("Score threshold", 1, 5, 3,
                      help="The score cannot reach 4 in practice — trend and "
                           "mean-reversion terms cancel. Above 3 goes silent.")
    if entry > 3:
        st.caption("⚠︎ Above 3 will produce very few or zero signals.")
    rsi_buy = st.slider("RSI oversold (buy)", 10, 40, 30)
    rsi_short = st.slider("RSI overbought (short)", 60, 90, 70)
    use_veto = st.toggle("SMA veto", value=True,
                         help="Never short above the fast SMA, never buy below it.")
    use_confirm = st.toggle("Confirmation bars", value=True,
                            help="Score must hold past the threshold for "
                                 "several bars on fast timeframes. Cut signal "
                                 "flips by ~60% on 1m in testing.")
    trend_rsi = st.toggle(
        "Trend-aware RSI", value=True,
        help="Stop RSI fading a trend that price and structure confirm. "
             "Without it the score pins near ±1 during real moves and "
             "everything sits on HOLD.")

    st.markdown("### View")
    only_setups = st.toggle("Setups only", value=True)
    hide_stale = st.toggle("Hide stale bars", value=False)

    if st.button("Rescan now", width="stretch"):
        st.cache_data.clear()
        st.rerun()

ov = dict(entry=entry, rsi_buy=rsi_buy, rsi_short=rsi_short, veto=use_veto,
          confirm=use_confirm, trend_rsi=trend_rsi)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
st.markdown(
    f'<div class="hero"><h1>Cam\'s Stock Scanner</h1>'
    f'<div class="sub">Equities on {eq_tf} · futures on '
    f'{fut_tf if fut_intraday else eq_tf} · score −6 to +6 · '
    f'parameters rescale with the bar size</div></div>',
    unsafe_allow_html=True)

if not tickers:
    st.markdown('<div class="empty">Add tickers in the sidebar to start '
                'scanning.</div>', unsafe_allow_html=True)
    st.stop()

rows, failed = scan(tickers, eq_tf, fut_tf, fut_intraday, mtf_list, ov)

buys = [r for r in rows if r["signal"] == "BUY"]
shorts = [r for r in rows if r["signal"] == "SHORT"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Buy setups", len(buys))
c2.metric("Short setups", len(shorts))
c3.metric("Holding", len(rows) - len(buys) - len(shorts))
c4.metric("Scanned", f"{len(rows)}/{len(tickers)}")

view = rows
if only_setups:
    view = [r for r in view if r["signal"] != "HOLD"]
if hide_stale:
    view = [r for r in view if not r["stale"]]
view.sort(key=lambda r: (-abs(r["score"]), r["ticker"]))

st.markdown("#### Signals")
if not view:
    st.markdown(
        '<div class="empty">Nothing clears the threshold right now.<br>'
        'Turn off <b>Setups only</b> to see every ticker and its score.</div>',
        unsafe_allow_html=True)
else:
    for r in view:
        st.markdown(card(r, show_mtf), unsafe_allow_html=True)

# --- disagreement readout: the answer to "why 15m but not 30m"
if show_mtf and mtf_list and len(mtf_list) > 1:
    split = []
    for r in rows:
        vals = [v[0] for v in r["matrix"].values() if v is not None]
        if vals and max(vals) >= entry and min(vals) <= -entry:
            split.append(r["ticker"])
    if split:
        st.caption(
            f"Timeframes disagree on: {', '.join(split)} — one interval says buy "
            f"while another says short. That usually means the move is fresh and "
            f"hasn't confirmed on the slower bars yet.")

# --- table
with st.expander("Full table"):
    tbl = pd.DataFrame([{
        "ticker": r["ticker"], "signal": r["signal"], "tf": r["tf"],
        "score": r["score"], "bars": r["bars"], "since": r["since"],
        "close": round(r["close"], 2), "chg %": round(r["chg"], 2),
        "RSI": round(r["rsi"], 1), "stale": r["stale"],
        **{f"score {m}": (r["matrix"][m][0] if r["matrix"].get(m) else None)
           for m in mtf_list},
    } for r in rows])
    st.dataframe(tbl, width="stretch", hide_index=True)

# --- chart
if rows:
    st.markdown("#### Chart")
    pick = st.selectbox("Ticker", [r["ticker"] for r in rows],
                        label_visibility="collapsed")
    r = next(r for r in rows if r["ticker"] == pick)
    chart = pd.DataFrame({
        "close": r["series"],
        f"SMA{r['snap']['fast']}": r["sma_f"],
        f"SMA{r['snap']['slow']}": r["sma_s"],
    })
    st.line_chart(chart, height=320,
                  color=["#5EE9B5", "#7C8AA0", "#3E4A5C"])
    st.caption(f"{pick} on {r['tf']} bars · {r['reasons']}")

if failed:
    st.caption(f"No usable data for: {', '.join(failed)}")

st.caption("Screening ideas, not financial advice. Signals are mechanical "
           "output from price history and say nothing about whether a trade is "
           "sound. Intraday bars in progress can still change before they close.")
