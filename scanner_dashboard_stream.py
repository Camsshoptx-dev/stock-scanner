"""
CAM'S STOCK SCANNER — STREAM SERVER
====================================
Single file. Needs only pandas, numpy and yfinance.

Run:  python scanner_dashboard_stream.py
  Board:         http://localhost:8080/board     (split scanner <-> charts, 30s flip)
  Combined box:  http://localhost:8080/stream    (signals + chart grid, for capture)
  Signals box:   http://localhost:8080/signals
  Trades box:    http://localhost:8080/trades
  Trades edit:   http://localhost:8080/trades/edit
  Force rescan:  http://localhost:8080/rescan

SCORING (-6 to +6, seven signed terms)
  RSI oversold / overbought      +2 / -2
  MACD vs signal line            +1 / -1
  SMA20 vs SMA50 (trend)         +1 / -1
  Price vs SMA20 (location)      +1 / -1
  Bollinger band touch           +1 / -1

Two filters run on top of the score:
  SMA20 veto — never short above the 20 SMA, never buy below it.
  Expiry     — drop any signal that has run past MAX_AGE consecutive bars.

Equities scan on daily bars. Futures scan on 15-min bars, because a daily
20/50 SMA can never stay in sync with an intraday scalp. Cards flag STALE
(last bar older than the current session) and ROLL (a jump in the continuous
futures series that looks like a contract splice rather than a real move).

Ctrl+C to stop.
"""

import html
import json
import os
import threading
import time
from datetime import datetime, date
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG — everything worth tweaking lives here
# ----------------------------------------------------------------------
PORT = 8080
SCAN_INTERVAL = 300         # rescan every 5 min
PAGE_REFRESH = 60           # each page hard-reloads this often (seconds)
CHART_SLOTS = 6             # charts on screen at once in /stream
CHART_ROTATE = 30           # rotate to the next set every N seconds

# --- /board : full-screen page that flips between the split scanner and charts
BOARD_PHASE = 30            # seconds on screen per phase (scanner, then charts)
BOARD_SLOTS = 6             # chart tiles in the board's chart phase (3 x 2)
BOARD_MAX_ROWS = 9          # setup cards per column before "+N more"
BOARD_POLL = 30             # seconds between /api/signals pulls (no page reload)
BOARD_SPLIT = "timeframe"   # "timeframe" -> daily swings | intraday scalps
                            # "direction" -> longs | shorts

# --- options phase (board only — /stream and /signals are untouched)
OPTIONS_ON = True           # add a third phase to /board
OPTIONS_INTERVAL = 900      # refresh chains every 15 min (they are slow)
OPTIONS_TARGET_DELTA = 0.40 # pick the contract nearest this delta
OPTIONS_MIN_DTE = 21
OPTIONS_MAX_DTE = 60
OPTIONS_MIN_OI = 100        # skip anything you cannot get out of
OPTIONS_MAX_SPREAD = 30     # % of mid
OPTIONS_MAX_TICKERS = 8     # cap chain pulls per cycle
OPTIONS_SLOTS = 6           # cards on the options pane

USE_VETO = True             # SMA20 veto on/off
VETO_ON_ENTRY_ONLY = False  # True = veto can block a new signal but never kill a live one
ROLL_GAP = 0.02             # single-bar move that suggests a contract roll splice

# ----------------------------------------------------------------------
# TIMEFRAME PROFILES
# ----------------------------------------------------------------------
# One row per timeframe. Everything that should change when the bar size
# changes lives here, so switching 1d -> 1m rescales the whole engine
# instead of running daily-tuned settings on one-minute noise.
#
#   period      how far back to pull. yfinance caps intraday history:
#               1m -> 7d, 2m/5m/15m/30m/90m -> 60d, 1h -> 730d.
#               Asking for more than the cap returns an EMPTY frame.
#   fast/slow   SMA lengths. Shorter on fast bars (9/21 is the scalper's
#               pair) so trend flips inside the session instead of lagging.
#   rsi_n       RSI lookback. Shorter = more responsive, so the bands must
#               widen to compensate.
#   rsi_buy/    RSI 30 fires almost every hour on 1m bars. Pushing the
#   rsi_short   bands to 25/75 keeps "oversold" meaning something.
#   macd        (fast, slow, signal) EMA spans, scaled with the bar.
#   entry       |score| needed to OPEN. Keep this at 3 — see the note on
#               rsi_trend_aware below. Setting it to 4 makes the scanner
#               go permanently silent, because the score cannot reach 4.
#   rsi_trend_aware
#               The score mixes trend terms (MACD, SMA cross, price vs SMA)
#               with mean-reversion terms (RSI extreme, Bollinger touch).
#               In a real trend those fight each other: the trend terms hit
#               +3 while RSI pins overbought for -2, netting +1. Measured on
#               a clean uptrend the RSI term averages -1.98 against +1.97 of
#               trend, so the two cancel almost exactly and the score ceiling
#               collapses to about +1.
#               When True, the RSI penalty is suppressed while trend and
#               location AGREE — overbought stops being a reason to fade a
#               confirmed uptrend, and only counts when price is already
#               rolling over. Leave False to keep the original mean-reversion
#               behaviour on the slower timeframes.
#   exit        hysteresis floor. Once open the signal holds until score
#               decays past this. Wider gap = fewer whipsaw exits.
#   confirm     score must stay past `entry` for this many CONSECUTIVE
#               bars before a signal opens. This is the single biggest
#               noise filter on 1m/2m.
#   max_age     bars before a stale signal is force-expired. Roughly one
#               session's worth on each timeframe.
#   veto_buf    how far past the 20 SMA counts as a real break. 0.15% is
#               ~35 NQ points, meaningless on a 1m chart, so it tightens.
#   min_bars    bars required before the ticker is scanned at all.
#   stale_h     last bar older than this many hours -> flag STALE.
TF_PROFILES = {
    "1m":  dict(period="5d",   fast=9,  slow=21, rsi_n=7,  bb_n=20,
                rsi_buy=25, rsi_short=75, macd=(6, 13, 5), rsi_trend_aware=True,
                entry=3, exit=0, confirm=3, max_age=30,
                veto_buf=0.0004, min_bars=60,  stale_h=0.5),
    "2m":  dict(period="10d",  fast=9,  slow=21, rsi_n=9,  bb_n=20,
                rsi_buy=27, rsi_short=73, macd=(8, 17, 6), rsi_trend_aware=True,
                entry=3, exit=0, confirm=2, max_age=30,
                veto_buf=0.0006, min_bars=60,  stale_h=0.75),
    "5m":  dict(period="60d",  fast=12, slow=30, rsi_n=9,  bb_n=20,
                rsi_buy=27, rsi_short=73, macd=(8, 17, 6), rsi_trend_aware=True,
                entry=3, exit=0, confirm=2, max_age=36,
                veto_buf=0.0008, min_bars=70,  stale_h=1),
    "15m": dict(period="60d",  fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), rsi_trend_aware=True,
                entry=3, exit=0, confirm=1, max_age=26,
                veto_buf=0.0015, min_bars=60,  stale_h=2),
    "30m": dict(period="60d",  fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), rsi_trend_aware=False,
                entry=3, exit=0, confirm=1, max_age=20,
                veto_buf=0.0020, min_bars=60,  stale_h=3),
    "1h":  dict(period="180d", fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), rsi_trend_aware=False,
                entry=3, exit=1, confirm=0, max_age=16,
                veto_buf=0.0025, min_bars=60,  stale_h=4),
    "1d":  dict(period="1y",   fast=20, slow=50, rsi_n=14, bb_n=20,
                rsi_buy=30, rsi_short=70, macd=(12, 26, 9), rsi_trend_aware=False,
                entry=3, exit=1, confirm=0, max_age=10,
                veto_buf=0.0015, min_bars=60,  stale_h=48),
}
TF_ORDER = ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]

# Live selection — change from the header buttons or /tf?scope=fut&v=5m
EQUITY_INTERVAL = "1d"
FUTURES_INTERVAL = "15m"
FUTURES_INTRADAY = True     # False = scan futures on the equity timeframe


def prof(interval):
    """Profile for a timeframe, falling back to 15m if something odd shows up."""
    return TF_PROFILES.get(interval, TF_PROFILES["15m"])


def tf_for(ticker):
    """Which timeframe this ticker gets scanned on."""
    if is_future(ticker) and FUTURES_INTRADAY:
        return FUTURES_INTERVAL
    return EQUITY_INTERVAL


# Legacy aliases. The render code further down still prints THRESHOLD and
# MAX_AGE in footers; keep them pointed at whatever the futures pane is
# currently using so the caption never contradicts the table.
def _sync_legacy():
    global RSI_BUY, RSI_SHORT, THRESHOLD, MAX_AGE, VETO_BUFFER
    global EQUITY_PERIOD, FUTURES_PERIOD, FAST, SLOW, RSI_N, BB_N, MIN_BARS
    p = prof(FUTURES_INTERVAL if FUTURES_INTRADAY else EQUITY_INTERVAL)
    e = prof(EQUITY_INTERVAL)
    RSI_BUY, RSI_SHORT = p["rsi_buy"], p["rsi_short"]
    THRESHOLD, MAX_AGE = p["entry"], p["max_age"]
    VETO_BUFFER = p["veto_buf"]
    FAST, SLOW, RSI_N, BB_N = p["fast"], p["slow"], p["rsi_n"], p["bb_n"]
    MIN_BARS = p["min_bars"]
    EQUITY_PERIOD, FUTURES_PERIOD = e["period"], p["period"]


RSI_BUY = RSI_SHORT = THRESHOLD = MAX_AGE = VETO_BUFFER = None
EQUITY_PERIOD = FUTURES_PERIOD = None
FAST = SLOW = RSI_N = BB_N = MIN_BARS = None
_sync_legacy()

TICKERS = [t.strip().upper() for t in (
    "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,"
    "SPY,QQQ,PLTR,COIN,SOFI,DIS,BA,UBER,SHOP,INTC,"
    "ES=F,MES=F,NQ=F,MNQ=F,YM=F,RTY=F"
).split(",") if t.strip()]

# CME futures are blocked in TradingView's free widgets -> chart CFD equivalents.
# Signals and prices still come from real futures data via yfinance.
TV_SYMBOL_MAP = {
    "ES=F":  "BLACKBULL:SPX500",
    "MES=F": "BLACKBULL:SPX500",
    "NQ=F":  "BLACKBULL:NAS100",
    "MNQ=F": "BLACKBULL:NAS100",
    "YM=F":  "BLACKBULL:US30",
    "RTY=F": "BLACKBULL:US2000",
}
TV_INTERVAL = {"1d": "D", "1h": "60", "30m": "30", "15m": "15",
               "5m": "5", "2m": "2", "1m": "1"}

TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.json")

try:
    import options_scanner
    OPTIONS_OK = True
except ImportError:
    OPTIONS_OK = False
    print("  note: options_scanner.py not found — options phase disabled.")

STATE = {"rows": [], "failed": [], "last_scan": None, "scanning": False,
         "options": [], "last_options": None, "opt_scanning": False}
STATE_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def is_future(ticker):
    return ticker.endswith("=F")


def download(symbols, period, interval):
    if not symbols:
        return None
    try:
        return yf.download(list(symbols), period=period, interval=interval,
                           progress=False, auto_adjust=True, group_by="ticker")
    except Exception as e:
        print(f"  download failed ({interval}): {e}")
        return None


def extract_close(raw, ticker):
    """Pull one Close series out of whatever shape yfinance returned."""
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
    tail = close.tail(lookback)
    return bool((tail.pct_change().abs() > ROLL_GAP).any())


# ----------------------------------------------------------------------
# INDICATORS + SCORING
# ----------------------------------------------------------------------
def indicators(close, p):
    """All indicators built from the timeframe profile, not globals."""
    n = p["rsi_n"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    mid = close.rolling(p["bb_n"]).mean()
    std = close.rolling(p["bb_n"]).std()
    m_fast, m_slow, m_sig = p["macd"]
    macd = (close.ewm(span=m_fast, adjust=False).mean()
            - close.ewm(span=m_slow, adjust=False).mean())
    return {
        "rsi": rsi,
        "sma_f": close.rolling(p["fast"]).mean(),
        "sma_s": close.rolling(p["slow"]).mean(),
        "macd": macd,
        "macd_sig": macd.ewm(span=m_sig, adjust=False).mean(),
        "bb_up": mid + 2 * std,
        "bb_lo": mid - 2 * std,
    }


def score_series(close, ind, p):
    def pm(cond):                       # +1 when true, -1 when false
        return cond.astype(int) * 2 - 1

    up = pm(ind["macd"] > ind["macd_sig"])
    trend = pm(ind["sma_f"] > ind["sma_s"])
    loc = pm(close > ind["sma_f"])

    over = ind["rsi"] > p["rsi_short"]
    under = ind["rsi"] < p["rsi_buy"]

    if p.get("rsi_trend_aware"):
        # Don't fade strength that trend AND location both confirm — that is
        # the cancellation that flattens the score to +/-1 in a real move.
        bull = (trend > 0) & (loc > 0)
        bear = (trend < 0) & (loc < 0)
        over = over & ~bull
        under = under & ~bear

    return (
        under.astype(int) * 2
        - over.astype(int) * 2
        + up + trend + loc
        + (close <= ind["bb_lo"]).astype(int)
        - (close >= ind["bb_up"]).astype(int)
    )


def build_signals(close, interval):
    """Walk the bars with state, so entry and exit can use different rules.

    Opening needs |score| >= entry for `confirm` consecutive bars. Once open,
    the signal survives until the score decays past exit, or price breaks the
    fast SMA by veto_buf. A vectorised threshold cannot express that, hence
    the loop.

    Every number comes from the timeframe profile, so a 1m scan runs 9/21
    SMAs, RSI(7) with 25/75 bands and a 2-bar confirmation, while a 1d scan
    keeps the original 20/50 + RSI(14) behaviour.
    """
    p = prof(interval)
    ind = indicators(close, p)
    score = score_series(close, ind, p)

    entry_thr, exit_thr = p["entry"], p["exit"]
    max_age, confirm = p["max_age"], p["confirm"]
    veto_buf = p["veto_buf"]

    sma_f = ind["sma_f"]
    states, vetoes, expiries = [], [], []
    state, run = "HOLD", 0
    streak_up, streak_dn = 0, 0     # consecutive bars past the entry threshold

    for i in range(len(close)):
        s_i = score.iloc[i]
        px = float(close.iloc[i])
        sma = sma_f.iloc[i]
        have_sma = pd.notna(sma)
        vetoed = False

        # a real break of the fast SMA, not a one-tick poke
        below = have_sma and px < sma * (1 - veto_buf)
        above = have_sma and px > sma * (1 + veto_buf)

        streak_up = streak_up + 1 if s_i >= entry_thr else 0
        streak_dn = streak_dn + 1 if s_i <= -entry_thr else 0

        if state == "HOLD":
            run = 0
            if streak_up > confirm:
                if USE_VETO and below:
                    vetoed = True
                else:
                    state, run = "BUY", 1
            elif streak_dn > confirm:
                if USE_VETO and above:
                    vetoed = True
                else:
                    state, run = "SHORT", 1

        elif state == "BUY":
            kill_veto = USE_VETO and not VETO_ON_ENTRY_ONLY and below
            if s_i < exit_thr or kill_veto:
                vetoed = kill_veto
                state, run = "HOLD", 0
            else:
                run += 1

        elif state == "SHORT":
            kill_veto = USE_VETO and not VETO_ON_ENTRY_ONLY and above
            if s_i > -exit_thr or kill_veto:
                vetoed = kill_veto
                state, run = "HOLD", 0
            else:
                run += 1

        expired = False
        if state != "HOLD" and run > max_age:
            state, run, expired = "HOLD", 0, True

        states.append(state)
        vetoes.append(vetoed)
        expiries.append(expired)

    sig = pd.Series(states, index=close.index)
    vetoed_s = pd.Series(vetoes, index=close.index)
    expired_s = pd.Series(expiries, index=close.index)

    snap = {k: float(v.iloc[-1]) for k, v in ind.items()}
    snap["score"] = float(score.iloc[-1])
    snap["vetoed"] = bool(vetoed_s.iloc[-1])
    snap["expired"] = bool(expired_s.iloc[-1])
    snap["entry_thr"] = entry_thr
    snap["exit_thr"] = exit_thr
    snap["confirm"] = confirm
    snap["tf"] = interval
    snap["rsi_buy"] = p["rsi_buy"]
    snap["rsi_short"] = p["rsi_short"]
    snap["fast"] = p["fast"]
    snap["rsi_trend_aware"] = bool(p.get("rsi_trend_aware"))

    valid = ind["sma_s"].notna()
    return sig[valid], close[valid], snap


def reasons_text(snap, last):
    why = [f"TF {snap.get('tf', '?')}"]
    r = snap["rsi"]
    lo, hi = snap.get("rsi_buy", 30), snap.get("rsi_short", 70)
    if r < lo:
        why.append(f"RSI {r:.0f} oversold (<{lo})")
    elif r > hi:
        why.append(f"RSI {r:.0f} overbought (>{hi})")
    else:
        why.append(f"RSI {r:.0f}")
    why.append("MACD bullish" if snap["macd"] > snap["macd_sig"] else "MACD bearish")
    why.append("Uptrend" if snap["sma_f"] > snap["sma_s"] else "Downtrend")
    fast = snap.get("fast", 20)
    why.append(f"Above {fast} SMA" if last > snap["sma_f"] else f"Below {fast} SMA")
    if last <= snap["bb_lo"]:
        why.append("Lower Bollinger")
    elif last >= snap["bb_up"]:
        why.append("Upper Bollinger")
    if snap["vetoed"]:
        why.append("VETOED by SMA rule")
    if snap["expired"]:
        why.append("EXPIRED past max bars")
    if snap.get("rsi_trend_aware"):
        why.append("RSI trend-aware")
    if snap.get("confirm"):
        why.append(f"needs {snap['confirm'] + 1} bars to confirm")
    if "exit_thr" in snap:
        why.append(f"holds while score beyond {snap['exit_thr']:+.0f}")
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


# ----------------------------------------------------------------------
# SCAN
# ----------------------------------------------------------------------
def run_scan():
    with STATE_LOCK:
        STATE["scanning"] = True

    # Group tickers by the timeframe they should be scanned on, then do one
    # download per group. period MUST travel with interval — asking for 1y of
    # 1m bars returns an empty frame and every ticker silently "fails".
    groups = {}
    for t in TICKERS:
        groups.setdefault(tf_for(t), []).append(t)

    sources = {}
    for interval, syms in groups.items():
        p = prof(interval)
        raw = download(syms, p["period"], interval)
        for t in syms:
            sources[t] = (raw, interval)

    rows, failed = [], []
    for t in TICKERS:
        raw, interval = sources.get(t, (None, EQUITY_INTERVAL))
        p = prof(interval)
        close = extract_close(raw, t)
        if close is None or len(close) < p["min_bars"]:
            failed.append(t)
            continue
        try:
            sig, cv, snap = build_signals(close, interval)
            if sig.empty or len(cv) < 2:
                failed.append(t)
                continue

            bars, since_ts, entry, chg_since = signal_age(sig, cv)
            last, prev = float(cv.iloc[-1]), float(cv.iloc[-2])
            age_h = bar_age_hours(cv.index[-1])
            fmt = "%m-%d" if interval == "1d" else "%m-%d %H:%M"

            rows.append({
                "ticker": t,
                "signal": sig.iloc[-1],
                "tf": interval,
                "score": snap["score"],
                "bars": bars,
                "since": f"{since_ts:{fmt}}",
                "entry": entry,
                "chg_since": chg_since,
                "close": last,
                "chg_1bar": (last / prev - 1) * 100,
                "rsi": snap["rsi"],
                "sma20": snap["sma_f"],
                "stale": age_h > p["stale_h"],
                "bar_age_h": age_h,
                "roll_gap": is_future(t) and has_roll_gap(cv),
                "reasons": reasons_text(snap, last),
            })
        except Exception as e:
            print(f"  {t} failed: {e}")
            failed.append(t)

    with STATE_LOCK:
        STATE["rows"] = rows
        STATE["failed"] = failed
        STATE["last_scan"] = datetime.now().strftime("%H:%M:%S")
        STATE["scanning"] = False

    setups = [r for r in rows if r["signal"] != "HOLD"]
    summary = ", ".join(f'{r["ticker"]}({r["signal"]} {r["score"]:+.0f})' for r in setups)
    tfs = "/".join(f"{k}:{len(v)}" for k, v in sorted(groups.items()))
    print(f'[{STATE["last_scan"]}] scan ok ({len(rows)} tickers [{tfs}], '
          f'{len(setups)} setups) -> {summary or "none"}')
    stale = [r["ticker"] for r in rows if r["stale"]]
    if stale:
        print(f'  stale bars: {", ".join(stale)}')
    if failed:
        print(f'  no data for: {", ".join(failed)}')


def scan_loop():
    while True:
        run_scan()
        time.sleep(SCAN_INTERVAL)


# ----------------------------------------------------------------------
# OPTIONS — prices whatever the scanner is currently calling
# ----------------------------------------------------------------------
def run_options_scan():
    """Pull chains only for equities that already have a live setup.

    Nothing here generates a signal. It answers one question about signals
    that already exist: what would it cost to express this with options,
    and what are the odds the market is giving that bet.
    """
    if not (OPTIONS_OK and OPTIONS_ON):
        return

    with STATE_LOCK:
        rows = list(STATE["rows"])
        STATE["opt_scanning"] = True

    setups = [r for r in rows if r["signal"] != "HOLD" and not is_future(r["ticker"])]
    setups.sort(key=lambda r: -abs(r["score"]))
    setups = setups[:OPTIONS_MAX_TICKERS]

    if not setups:
        with STATE_LOCK:
            STATE["options"] = []
            STATE["last_options"] = datetime.now().strftime("%H:%M:%S")
            STATE["opt_scanning"] = False
        print("  [options] no equity setups to price")
        return

    smap = {r["ticker"]: r["signal"] for r in setups}
    try:
        opts = options_scanner.best_contracts(
            list(smap.keys()), signal_map=smap,
            target_delta=OPTIONS_TARGET_DELTA,
            min_dte=OPTIONS_MIN_DTE, max_dte=OPTIONS_MAX_DTE,
            min_open_interest=OPTIONS_MIN_OI,
            max_spread_pct=OPTIONS_MAX_SPREAD)
    except Exception as e:
        print(f"  [options] scan failed: {e}")
        opts = []

    with STATE_LOCK:
        STATE["options"] = opts
        STATE["last_options"] = datetime.now().strftime("%H:%M:%S")
        STATE["opt_scanning"] = False

    if opts:
        brief = ", ".join(f'{o["ticker"]} {o["prob_itm"]:.0f}%' for o in opts)
        print(f'[{STATE["last_options"]}] options ok ({len(opts)} contracts) -> {brief}')
    else:
        print("  [options] nothing cleared the liquidity filters")


def options_loop():
    time.sleep(35)          # let the first price scan land first
    while True:
        run_options_scan()
        time.sleep(OPTIONS_INTERVAL)


# ----------------------------------------------------------------------
# TRADES (paper P&L, persisted to trades.json)
# ----------------------------------------------------------------------
def load_trades():
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root { --accent:#3DD9B0; --bg:#0B0F17; --card:#111726; --card2:#161C2A;
        --line:#1C2536; --text:#F4F6FB; --muted:#8A94A6; --dim:#5F6B7E;
        --buy-bg:#0F2A20; --buy:#4FD1A0; --short-bg:#2A1315; --short:#F0837F;
        --warn-bg:#2A2413; --warn:#E0B84F; }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'Inter',sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:2rem 20px; }
h1 { font-size:1.5rem; font-weight:600; letter-spacing:0.01em;
     display:flex; align-items:center; gap:10px; }
h1::before { content:"S"; display:inline-flex; align-items:center; justify-content:center;
     width:30px; height:30px; border-radius:8px; background:var(--accent);
     color:var(--bg); font-size:17px; font-weight:600; }
.sub { color:var(--dim); font-size:13px; margin:4px 0 20px 40px; }
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
.setup .tkr { color:var(--text); font-size:15px; font-weight:600; line-height:1.2;
     display:flex; align-items:center; gap:6px; }
.setup .meta { color:var(--dim); font-size:11px; margin-top:2px; }
.setup .price { color:var(--text); font-size:15px; text-align:right; line-height:1.2; }
.setup .chg { font-size:12px; text-align:right; }
.up { color:var(--buy); } .down { color:var(--short); }
.tag { font-size:10px; font-weight:600; padding:2px 6px; border-radius:4px;
     background:var(--warn-bg); color:var(--warn); }
.tf { font-size:10px; color:var(--dim); border:1px solid var(--line);
     padding:2px 6px; border-radius:4px; font-weight:500; }
.note { background:var(--card2); border-left:2px solid var(--warn); color:var(--muted);
     font-size:12px; padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:16px;
     line-height:1.5; }
.grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
.chart { background:var(--card); border:1px solid var(--line); border-radius:12px;
     overflow:hidden; height:300px; position:relative; }
.chart .tag2 { position:absolute; top:8px; left:8px; z-index:2; font-size:11px;
     font-weight:600; padding:3px 8px; border-radius:6px; background:var(--card2);
     color:var(--accent); border:1px solid var(--line); }
.chart iframe { width:100%; height:100%; border:0; }
.stream-wrap { display:grid; grid-template-columns:400px 1fr; gap:20px; }
.footer { color:var(--dim); font-size:12px; margin-top:16px; line-height:1.6; }
table { width:100%; border-collapse:collapse; margin-bottom:16px; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line);
     font-size:14px; }
th { color:var(--muted); font-size:11px; font-weight:600; letter-spacing:0.06em;
     text-transform:uppercase; }
input, select, button { background:var(--card2); color:var(--text);
     border:1px solid var(--line); border-radius:8px; padding:8px 10px;
     font-size:14px; font-family:'Inter',sans-serif; }
button { background:var(--buy-bg); color:var(--buy); font-weight:600; cursor:pointer; }
form.row { display:flex; gap:8px; flex-wrap:wrap; }
a { color:var(--accent); text-decoration:none; }
@media (max-width:900px) {
  .stream-wrap { grid-template-columns:1fr; }
  .grid { grid-template-columns:1fr; }
}

/* ---------- /board : full-screen two-phase display ---------- */
body.board { overflow:hidden; height:100vh; }
.bhead { position:fixed; top:0; left:0; right:0; height:58px; z-index:6;
     display:flex; align-items:center; gap:16px; padding:0 22px;
     background:var(--bg); border-bottom:1px solid var(--line); }
.bhead .logo { width:28px; height:28px; border-radius:8px; background:var(--accent);
     color:var(--bg); font-weight:600; font-size:16px; display:flex;
     align-items:center; justify-content:center; }
.bhead .name { font-size:16px; font-weight:600; }
.bhead .stat { color:var(--dim); font-size:12px; }
.bhead .spacer { flex:1; }
.bhead .phase { font-size:11px; font-weight:600; letter-spacing:0.1em;
     color:var(--accent); background:var(--card2); border:1px solid var(--line);
     padding:5px 10px; border-radius:6px; }
.bbar { position:fixed; top:57px; left:0; height:2px; width:0;
     background:var(--accent); z-index:7; }
.pane { position:fixed; top:58px; left:0; right:0; bottom:0; z-index:2;
     opacity:0; pointer-events:none; transition:opacity .4s ease; }
.pane.on { opacity:1; pointer-events:auto; z-index:3; }
.split { display:grid; grid-template-columns:1fr 1fr; height:100%; }
.col { padding:16px 20px; overflow:hidden; display:flex; flex-direction:column; }
.col + .col { border-left:1px solid var(--line); }
.col-head { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.col-head .ttl { font-size:13px; font-weight:600; letter-spacing:0.06em;
     text-transform:uppercase; }
.col-head .cnt { font-size:11px; color:var(--dim); border:1px solid var(--line);
     padding:2px 7px; border-radius:5px; }
.col-head .hint { color:var(--dim); font-size:11px; margin-left:auto; }
.col.swing .ttl { color:var(--accent); }
.col.scalp .ttl { color:var(--warn); }
.col .list { flex:1; overflow:hidden; }
.col .setup { margin-bottom:7px; padding:11px 14px; }
.empty { background:var(--card); border-radius:12px; padding:16px;
     color:var(--dim); font-size:13px; }
.more { color:var(--dim); font-size:11px; padding:4px 2px; }
.bgrid { display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:1fr 1fr;
     gap:10px; height:100%; padding:12px 16px 16px; }
.bgrid .chart { height:100%; }
.bfoot { position:fixed; bottom:0; left:0; right:0; z-index:6; padding:5px 22px;
     background:var(--bg); border-top:1px solid var(--line);
     color:var(--dim); font-size:11px; }
.pane { bottom:26px; }

/* ---------- options phase ---------- */
.ogrid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
     padding:14px 18px; height:100%; align-content:start; }
.ocard { background:var(--card); border:1px solid var(--line); border-radius:12px;
     padding:13px 15px; }
.ocard .otop { display:flex; align-items:flex-start; justify-content:space-between;
     margin-bottom:11px; gap:10px; }
.ocard .oname { color:var(--text); font-size:15px; font-weight:600; line-height:1.3; }
.ocard .oexp { color:var(--dim); font-size:11px; margin-top:2px; }
.ocard .ocost { color:var(--text); font-size:19px; font-weight:600; text-align:right;
     line-height:1.1; white-space:nowrap; }
.ocard .ocost small { display:block; color:var(--dim); font-size:10px; font-weight:400; }
.ocard .orow { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
.ocard .ocell { background:var(--card2); border-radius:7px; padding:7px 8px; }
.ocard .ok { color:var(--dim); font-size:9px; letter-spacing:0.05em;
     text-transform:uppercase; margin-bottom:2px; }
.ocard .ov { color:var(--text); font-size:13px; font-weight:600; }
.ocard .ov.good { color:var(--buy); }
.ocard .ov.mid { color:var(--warn); }
.ocard .ov.bad { color:var(--short); }
.ohead { padding:14px 20px 0; display:flex; align-items:center; gap:10px; }
.ohead .ttl { font-size:13px; font-weight:600; letter-spacing:0.06em;
     text-transform:uppercase; color:var(--accent); }
.ohead .cnt { font-size:11px; color:var(--dim); border:1px solid var(--line);
     padding:2px 7px; border-radius:5px; }
.ohead .hint { color:var(--dim); font-size:11px; margin-left:auto; }
"""


def refresh_script():
    # Hard reload with a cache-buster so the page can never go stale mid-stream.
    return f"""
<script>
setTimeout(function() {{
  var u = new URL(window.location.href);
  u.searchParams.set('t', Date.now());
  window.location.replace(u.toString());
}}, {PAGE_REFRESH * 1000});
</script>"""


TF_CSS = """
.tfrow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0 14px}
.tfl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#5F6B7E;
     margin-right:2px}
.tfl+.tfb{margin-left:0}
.tfb{display:inline-block;padding:4px 10px;border-radius:6px;font-size:12px;
     font-weight:600;text-decoration:none;color:#8A97A8;background:#141A24;
     border:1px solid #232C3A}
.tfb:hover{color:#C9D4E2;border-color:#3A4658}
.tfb.on{color:#0B0F14;background:#5EE9B5;border-color:#5EE9B5}
.tfn{flex-basis:100%;font-size:11px;color:#5F6B7E;margin-top:4px}
"""


def page(title, body, auto_refresh=True):
    tail = refresh_script() if auto_refresh else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store">
<title>{title}</title><style>{CSS}{TF_CSS}</style></head>
<body><div class="wrap">{body}</div>{tail}</body></html>"""


def tf_buttons(scope, current):
    out = []
    for tf in TF_ORDER:
        cls = "tfb on" if tf == current else "tfb"
        out.append(f'<a class="{cls}" href="/tf?scope={scope}&v={tf}">{tf}</a>')
    return "".join(out)


def header_html():
    with STATE_LOCK:
        last, scanning, n = STATE["last_scan"], STATE["scanning"], len(STATE["rows"])
    status = "scanning…" if scanning or not last else f"live {last}"
    fp = prof(FUTURES_INTERVAL)
    return (f'<h1>Stock Scanner</h1>'
            f'<div class="sub">{n} scanned &middot; {date.today():%a %d %b %Y}'
            f' &middot; {status} &middot; auto-refresh {PAGE_REFRESH}s</div>'
            f'<div class="tfrow"><span class="tfl">Equities</span>'
            f'{tf_buttons("eq", EQUITY_INTERVAL)}'
            f'<span class="tfl">Futures</span>'
            f'{tf_buttons("fut", FUTURES_INTERVAL)}'
            f'<span class="tfn">futures: {fp["fast"]}/{fp["slow"]} SMA &middot; '
            f'RSI({fp["rsi_n"]}) {fp["rsi_buy"]}/{fp["rsi_short"]} &middot; '
            f'entry &plusmn;{fp["entry"]} &middot; confirm {fp["confirm"] + 1} bar'
            f'{"s" if fp["confirm"] else ""} &middot; expire {fp["max_age"]}'
            f'{" &middot; RSI trend-aware" if fp.get("rsi_trend_aware") else ""}'
            f'</span></div>')


def metrics_html(rows):
    b = sum(1 for r in rows if r["signal"] == "BUY")
    s = sum(1 for r in rows if r["signal"] == "SHORT")
    h = len(rows) - b - s
    return (f'<div class="cardrow">'
            f'<div class="metric m-buy"><div class="lbl">Buy setups</div><div class="val">{b}</div></div>'
            f'<div class="metric m-short"><div class="lbl">Short setups</div><div class="val">{s}</div></div>'
            f'<div class="metric m-hold"><div class="lbl">Holding</div><div class="val">{h}</div></div>'
            f'</div>')


def notes_html(rows):
    out = ""
    stale = [r["ticker"] for r in rows if r["stale"]]
    if stale:
        shown = ", ".join(stale[:12]) + ("…" if len(stale) > 12 else "")
        out += (f'<div class="note">Last bar is older than the current session for '
                f'<b>{len(stale)}</b> ticker(s): {shown}. These signals sit on closed '
                f'bars and will not move until the next session prints.</div>')
    rolls = [r["ticker"] for r in rows if r["roll_gap"]]
    if rolls:
        out += (f'<div class="note">Possible contract roll splice in recent history for '
                f'{", ".join(rolls)}. Continuous futures series stitch contracts together '
                f'and the jump at the seam feeds MACD and Bollinger as a real move.</div>')
    return out


def setup_card(r):
    cls = "buy" if r["signal"] == "BUY" else "short"
    ch = r["chg_1bar"]
    ud, sign = ("up", "+") if ch >= 0 else ("down", "")
    tags = ('<span class="tag">STALE</span>' if r["stale"] else "") + \
           ('<span class="tag">ROLL</span>' if r["roll_gap"] else "")
    return (f'<div class="setup"><div class="left">'
            f'<span class="pill {cls}">{r["signal"]}</span>'
            f'<div><div class="tkr">{html.escape(r["ticker"])}'
            f'<span class="tf">{r["tf"]}</span>{tags}</div>'
            f'<div class="meta">score {r["score"]:+.0f} &middot; {r["bars"]} bars '
            f'&middot; since {r["since"]}</div></div></div>'
            f'<div><div class="price">{r["close"]:,.2f}</div>'
            f'<div class="chg {ud}">{sign}{ch:.2f}%</div></div></div>')


def setups_html(rows):
    order = {"BUY": 0, "SHORT": 1}
    setups = sorted((r for r in rows if r["signal"] != "HOLD"),
                    key=lambda r: (order[r["signal"]], -abs(r["score"])))
    head = '<div class="section">Setups</div>'
    if not setups:
        msg = ("First scan running — give it ~30 seconds…" if not rows
               else "No buy or short setups right now. Everything is holding.")
        return head + f'<div class="setup"><div class="meta">{msg}</div></div>'
    return head + "".join(setup_card(r) for r in setups)


def charts_html(rows):
    """6-slot grid rotating through every setup.

    Rotation position comes from the clock, not a counter, so the 60 s page
    reload never snaps it back to the first page.
    """
    order = {"BUY": 0, "SHORT": 1}
    setups = sorted((r for r in rows if r["signal"] != "HOLD"),
                    key=lambda r: (order[r["signal"]], -abs(r["score"])))

    seen, charts = set(), []
    for r in setups:
        sym = TV_SYMBOL_MAP.get(r["ticker"], r["ticker"])
        if sym in seen:
            continue
        seen.add(sym)
        charts.append({"t": r["ticker"], "s": sym, "a": r["signal"],
                       "i": TV_INTERVAL.get(r["tf"], "15")})
    if not charts:
        charts = [{"t": t, "s": TV_SYMBOL_MAP.get(t, t), "a": "HOLD",
                   "i": "15" if is_future(t) else "D"}
                  for t in ["ES=F", "NQ=F", "YM=F", "SPY", "QQQ", "AAPL"]]

    slots = "".join(
        f'<div class="chart"><span class="tag2" id="tag{i}"></span>'
        f'<iframe id="slot{i}" loading="lazy" src="about:blank"></iframe></div>'
        for i in range(CHART_SLOTS))

    return f"""<div class="grid">{slots}</div>
<script>
var CHARTS = {json.dumps(charts)};
var SLOTS = {CHART_SLOTS};
var ROTATE = {CHART_ROTATE * 1000};
var BASE = "https://s.tradingview.com/widgetembed/?symbol=SYM&interval=INT&theme=dark&style=1&hidetoptoolbar=1&hidelegend=1&saveimage=0&toolbarbg=0B0F17";
function showPage() {{
  var pages = Math.max(1, Math.ceil(CHARTS.length / SLOTS));
  var pg = Math.floor(Date.now() / ROTATE) % pages;
  for (var i = 0; i < SLOTS; i++) {{
    var c = CHARTS[pg * SLOTS + i];
    var f = document.getElementById('slot' + i);
    var tag = document.getElementById('tag' + i);
    if (!c) {{ f.src = 'about:blank'; f.removeAttribute('data-sym'); tag.textContent = ''; continue; }}
    var url = BASE.replace('SYM', encodeURIComponent(c.s)).replace('INT', c.i);
    if (f.getAttribute('data-sym') !== c.s + c.i) {{
      f.src = url;
      f.setAttribute('data-sym', c.s + c.i);
    }}
    tag.textContent = c.t + ' \\u00b7 ' + c.a;
  }}
}}
showPage();
setInterval(showPage, 1000);
</script>"""


DISCLAIMER = ('<div class="footer">Score runs -6 to +6 across RSI, MACD, SMA trend, '
              'price vs the 20 SMA, and Bollinger touches. Anything inside '
              f'&plusmn;{THRESHOLD} holds, shorts above the 20 SMA are vetoed, and '
              f'signals expire after {MAX_AGE} bars. '
              'Screening ideas, not financial advice.</div>')


# ----------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------
def render_signals():
    with STATE_LOCK:
        rows = list(STATE["rows"])
    return page("Stock Scanner — Signals",
                header_html() + metrics_html(rows) + notes_html(rows)
                + setups_html(rows) + DISCLAIMER)


def render_stream():
    with STATE_LOCK:
        rows = list(STATE["rows"])
    body = header_html() + f"""
<div class="stream-wrap">
  <div>{metrics_html(rows)}{notes_html(rows)}{setups_html(rows)}</div>
  <div>{charts_html(rows)}</div>
</div>{DISCLAIMER}"""
    return page("Stock Scanner — Stream", body)


BOARD_JS = r"""
var ROWS = [], CHARTS = [], OPTS = [];

function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
  });
}

/* Which column a setup belongs in. Timeframe split puts anything scanned on
   intraday bars on the scalp side; direction split uses BUY vs SHORT. */
function isScalp(r) {
  return CFG.split === 'direction' ? r.signal === 'SHORT' : r.tf !== '1d';
}

function isSetup(r) { return r.signal !== 'HOLD'; }

function sortSetups(list) {
  var order = {BUY: 0, SHORT: 1};
  return list.slice().sort(function (a, b) {
    return (order[a.signal] - order[b.signal]) || (Math.abs(b.score) - Math.abs(a.score));
  });
}

function num(v, d) {
  return Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
}

function card(r) {
  var cls = r.signal === 'BUY' ? 'buy' : 'short';
  var ch = Number(r.chg_1bar);
  var ud = ch >= 0 ? 'up' : 'down';
  var sg = ch >= 0 ? '+' : '';
  var tags = (r.stale ? '<span class="tag">STALE</span>' : '') +
             (r.roll_gap ? '<span class="tag">ROLL</span>' : '');
  var sc = (r.score >= 0 ? '+' : '') + Number(r.score).toFixed(0);
  return '<div class="setup"><div class="left">' +
    '<span class="pill ' + cls + '">' + esc(r.signal) + '</span>' +
    '<div><div class="tkr">' + esc(r.ticker) +
    '<span class="tf">' + esc(r.tf) + '</span>' + tags + '</div>' +
    '<div class="meta">score ' + sc + ' \u00b7 ' + esc(r.bars) + ' bars \u00b7 since ' +
    esc(r.since) + '</div></div></div>' +
    '<div><div class="price">' + num(r.close, 2) + '</div>' +
    '<div class="chg ' + ud + '">' + sg + num(ch, 2) + '%</div></div></div>';
}

function renderCol(listId, cntId, list) {
  var el = document.getElementById(listId);
  document.getElementById(cntId).textContent = list.length;
  if (!ROWS.length) {
    el.innerHTML = '<div class="empty">First scan running \u2014 give it ~30 seconds\u2026</div>';
    return;
  }
  if (!list.length) {
    el.innerHTML = '<div class="empty">Nothing qualifying on this side right now.</div>';
    return;
  }
  var shown = list.slice(0, CFG.maxRows);
  var out = shown.map(card).join('');
  if (list.length > shown.length) out += '<div class="more">+' + (list.length - shown.length) + ' more</div>';
  el.innerHTML = out;
}

function render() {
  var setups = sortSetups(ROWS.filter(isSetup));
  renderCol('listA', 'cntA', setups.filter(function (r) { return !isScalp(r); }));
  renderCol('listB', 'cntB', setups.filter(isScalp));

  var seen = {}, out = [];
  setups.forEach(function (r) {
    var sym = CFG.tvmap[r.ticker] || r.ticker;
    if (seen[sym]) return;
    seen[sym] = 1;
    out.push({t: r.ticker, s: sym, a: r.signal, i: CFG.tvint[r.tf] || '15'});
  });
  CHARTS = out.length ? out : CFG.fallback;
}

/* Only swap an iframe's src when the symbol actually changes, so a chart that
   is already on screen never reloads and flickers mid-stream. */
function paintCharts(pg) {
  for (var i = 0; i < CFG.slots; i++) {
    var c = CHARTS[pg * CFG.slots + i];
    var f = document.getElementById('bslot' + i);
    var tag = document.getElementById('btag' + i);
    if (!c) {
      f.src = 'about:blank';
      f.removeAttribute('data-sym');
      tag.textContent = '';
      continue;
    }
    var key = c.s + c.i;
    if (f.getAttribute('data-sym') !== key) {
      f.src = CFG.base.replace('SYM', encodeURIComponent(c.s)).replace('INT', c.i);
      f.setAttribute('data-sym', key);
    }
    tag.textContent = c.t + ' \u00b7 ' + c.a;
  }
}

/* ---------- options phase ---------- */

function cls(v, mid, bad, invert) {
  var x = Math.abs(v);
  if (invert) return x > bad ? 'bad' : (x > mid ? 'mid' : 'good');
  return x < bad ? 'bad' : (x < mid ? 'mid' : 'good');
}

function optCard(o) {
  var side = o.side === 'BUY' ? 'buy' : 'short';
  var pc = cls(o.prob_itm, 40, 20, false);
  var mc = cls(o.req_move, 5, 10, true);
  var sc = cls(o.spread_pct, 8, 15, true);
  return '<div class="ocard"><div class="otop"><div>' +
    '<div class="oname"><span class="pill ' + side + '">' + esc(o.side) + '</span> ' +
    esc(o.ticker) + ' $' + o.strike + ' ' + esc(o.kind.toUpperCase()) + '</div>' +
    '<div class="oexp">' + esc(o.expiration) + ' \u00b7 ' + o.dte + 'd \u00b7 IV ' +
    Number(o.iv).toFixed(0) + '%</div></div>' +
    '<div class="ocost">$' + num(o.cost, 0) + '<small>per contract</small></div>' +
    '</div><div class="orow">' +
    '<div class="ocell"><div class="ok">Prob ITM</div><div class="ov ' + pc + '">' +
      Number(o.prob_itm).toFixed(0) + '%</div></div>' +
    '<div class="ocell"><div class="ok">Move</div><div class="ov ' + mc + '">' +
      (o.req_move >= 0 ? '+' : '') + Number(o.req_move).toFixed(1) + '%</div></div>' +
    '<div class="ocell"><div class="ok">B/E</div><div class="ov">$' +
      num(o.breakeven, 2) + '</div></div>' +
    '<div class="ocell"><div class="ok">Spread</div><div class="ov ' + sc + '">' +
      Number(o.spread_pct).toFixed(0) + '%</div></div>' +
    '</div></div>';
}

function renderOpts() {
  var el = document.getElementById('ogrid');
  if (!el) return;
  document.getElementById('ocnt').textContent = OPTS.length;
  if (!OPTS.length) {
    el.innerHTML = '<div class="empty">No liquid contracts on the current setups. ' +
      'Most chains fail the filters \u2014 that is the normal result.</div>';
    return;
  }
  el.innerHTML = OPTS.slice(0, CFG.oslots).map(optCard).join('');
}

function loadOptions() {
  fetch('/api/options?t=' + Date.now())
    .then(function (r) { return r.json(); })
    .then(function (d) { OPTS = d.options || []; renderOpts(); })
    .catch(function () { /* keep the last good options on screen */ });
}

/* Phase comes off the wall clock, not a counter, so nothing drifts and a
   reload lands back on the same rhythm. */
function tick() {
  var span = CFG.phase * 1000;
  var now = Date.now();
  var cycle = Math.floor(now / span);
  var n = CFG.phases.length;
  var idx = cycle % n;
  var name = CFG.phases[idx];

  document.getElementById('paneScan').classList.toggle('on', name === 'SCANNER');
  document.getElementById('paneChart').classList.toggle('on', name === 'CHARTS');
  var po = document.getElementById('paneOpt');
  if (po) po.classList.toggle('on', name === 'OPTIONS');

  document.getElementById('phase').textContent = name;
  document.getElementById('bar').style.width = (((now % span) / span) * 100).toFixed(2) + '%';

  var pages = Math.max(1, Math.ceil(CHARTS.length / CFG.slots));
  paintCharts(Math.floor(cycle / n) % pages);
}

function load() {
  fetch('/api/signals?t=' + Date.now())
    .then(function (r) { return r.json(); })
    .then(function (d) {
      ROWS = d.rows || [];
      var n = ROWS.length;
      var setups = ROWS.filter(isSetup).length;
      document.getElementById('stat').textContent =
        n + ' scanned \u00b7 ' + setups + ' setups \u00b7 last scan ' + (d.last_scan || '\u2026');
      render();
    })
    .catch(function () { /* keep the last good board on screen */ });
}

load();
render();
tick();
setInterval(load, CFG.poll * 1000);
setInterval(tick, 250);

if (CFG.options) {
  loadOptions();
  setInterval(loadOptions, CFG.opoll * 1000);
}
"""


def render_board():
    """Full-screen display: split scanner for BOARD_PHASE seconds, then charts.

    The whole thing runs off /api/signals in the browser, so the page never
    hard-reloads and the TradingView iframes stay warm behind the scanner.
    """
    by_tf = BOARD_SPLIT != "direction"
    left_title = "Swing / position" if by_tf else "Long setups"
    right_title = "Scalps" if by_tf else "Short setups"
    left_hint = "daily bars" if by_tf else "BUY side"
    right_hint = f"{FUTURES_INTERVAL} futures" if by_tf else "SHORT side"

    fallback = [{"t": t, "s": TV_SYMBOL_MAP.get(t, t), "a": "HOLD",
                 "i": "15" if is_future(t) else "D"}
                for t in ["ES=F", "NQ=F", "YM=F", "SPY", "QQQ", "AAPL"]]

    opts_on = OPTIONS_OK and OPTIONS_ON
    phases = ["SCANNER", "CHARTS"] + (["OPTIONS"] if opts_on else [])

    cfg = {
        "phase": BOARD_PHASE,
        "slots": BOARD_SLOTS,
        "maxRows": BOARD_MAX_ROWS,
        "poll": BOARD_POLL,
        "split": BOARD_SPLIT,
        "phases": phases,
        "options": opts_on,
        "oslots": OPTIONS_SLOTS,
        "opoll": 60,
        "tvmap": TV_SYMBOL_MAP,
        "tvint": TV_INTERVAL,
        "fallback": fallback,
        "base": ("https://s.tradingview.com/widgetembed/?symbol=SYM&interval=INT"
                 "&theme=dark&style=1&hidetoptoolbar=1&hidelegend=1&saveimage=0"
                 "&toolbarbg=0B0F17"),
    }

    tiles = "".join(
        f'<div class="chart"><span class="tag2" id="btag{i}"></span>'
        f'<iframe id="bslot{i}" loading="eager" src="about:blank"></iframe></div>'
        for i in range(BOARD_SLOTS))

    opt_pane = f"""
<div class="pane" id="paneOpt">
  <div class="ohead"><span class="ttl">Options on live setups</span>
    <span class="cnt" id="ocnt">0</span>
    <span class="hint">nearest {OPTIONS_TARGET_DELTA:.2f} delta &middot;
      {OPTIONS_MIN_DTE}-{OPTIONS_MAX_DTE} DTE &middot; OI &ge; {OPTIONS_MIN_OI}</span></div>
  <div class="ogrid" id="ogrid"></div>
</div>""" if opts_on else ""

    foot_opt = (" &middot; options priced off chain IV, not a recommendation"
                if opts_on else "")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store">
<title>Stock Scanner — Board</title><style>{CSS}</style></head>
<body class="board">

<div class="bhead">
  <div class="logo">S</div>
  <div class="name">Stock Scanner</div>
  <div class="stat" id="stat">loading…</div>
  <div class="spacer"></div>
  <div class="phase" id="phase">SCANNER</div>
</div>
<div class="bbar" id="bar"></div>

<div class="pane on" id="paneScan">
  <div class="split">
    <div class="col swing">
      <div class="col-head"><span class="ttl">{left_title}</span>
        <span class="cnt" id="cntA">0</span>
        <span class="hint">{left_hint}</span></div>
      <div class="list" id="listA"></div>
    </div>
    <div class="col scalp">
      <div class="col-head"><span class="ttl">{right_title}</span>
        <span class="cnt" id="cntB">0</span>
        <span class="hint">{right_hint}</span></div>
      <div class="list" id="listB"></div>
    </div>
  </div>
</div>

<div class="pane" id="paneChart">
  <div class="bgrid">{tiles}</div>
</div>
{opt_pane}

<div class="bfoot">Flips every {BOARD_PHASE}s &middot; score -6 to +6 &middot;
shorts above the 20 SMA vetoed &middot; signals expire after {MAX_AGE} bars{foot_opt} &middot;
screening ideas, not financial advice.</div>

<script>var CFG = {json.dumps(cfg)};</script>
<script>{BOARD_JS}</script>
</body></html>"""


def render_trades(edit=False):
    trades = load_trades()
    total = sum(t.get("pnl", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    wr = wins / len(trades) * 100 if trades else 0.0
    tcls = "up" if total >= 0 else "down"

    rows_html = ""
    for i, t in enumerate(trades):
        pcls = "up" if t.get("pnl", 0) >= 0 else "down"
        delete = f'<a href="/trades/delete?i={i}">&#10005;</a>' if edit else ""
        rows_html += (f'<tr><td>{html.escape(str(t.get("date","")))}</td>'
                      f'<td>{html.escape(str(t.get("ticker","")))}</td>'
                      f'<td>{html.escape(str(t.get("side","")))}</td>'
                      f'<td class="{pcls}">{t.get("pnl",0):+,.2f}</td><td>{delete}</td></tr>')
    if not rows_html:
        rows_html = '<tr><td colspan="5" style="color:#5F6B7E">No trades logged yet.</td></tr>'

    form = """
<form class="row" method="GET" action="/trades/add">
  <input name="ticker" placeholder="Ticker" required>
  <select name="side"><option>LONG</option><option>SHORT</option></select>
  <input name="pnl" placeholder="P&amp;L e.g. 125 or -40" required>
  <button type="submit">Add trade</button>
</form>""" if edit else ""

    body = header_html() + f"""
<div class="cardrow">
  <div class="metric m-hold"><div class="lbl">Total P&amp;L</div>
    <div class="val {tcls}">{total:+,.2f}</div></div>
  <div class="metric m-hold"><div class="lbl">Trades logged</div>
    <div class="val">{len(trades)}</div></div>
  <div class="metric m-hold"><div class="lbl">Win rate</div>
    <div class="val">{wr:.0f}%</div></div>
</div>
<div class="section">Trade log</div>
<table><tr><th>Date</th><th>Ticker</th><th>Side</th><th>P&amp;L</th><th></th></tr>
{rows_html}</table>
{form}
<div class="footer">Paper trades &middot; educational only &middot; not financial advice.</div>"""
    # no auto-refresh on the edit page so the form doesn't reload under you
    return page("Stock Scanner — Trades", body, auto_refresh=not edit)


# ----------------------------------------------------------------------
# HTTP SERVER
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, content, ctype="text/html"):
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, where):
        self.send_response(302)
        self.send_header("Location", where)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        q = parse_qs(parsed.query)

        if path == "/":
            self._redirect("/stream")
        elif path == "/board":
            self._send(render_board())
        elif path == "/stream":
            self._send(render_stream())
        elif path == "/signals":
            self._send(render_signals())
        elif path == "/trades":
            self._send(render_trades(edit=False))
        elif path == "/trades/edit":
            self._send(render_trades(edit=True))
        elif path == "/trades/add":
            trades = load_trades()
            try:
                trades.append({
                    "date": datetime.now().strftime("%m-%d"),
                    "ticker": q.get("ticker", [""])[0].upper()[:8],
                    "side": q.get("side", ["LONG"])[0],
                    "pnl": float(q.get("pnl", ["0"])[0]),
                })
                save_trades(trades)
            except Exception as e:
                print(f"[trades] add failed: {e}")
            self._redirect("/trades/edit")
        elif path == "/trades/delete":
            trades = load_trades()
            try:
                idx = int(q.get("i", ["-1"])[0])
                if 0 <= idx < len(trades):
                    trades.pop(idx)
                    save_trades(trades)
            except Exception as e:
                print(f"[trades] delete failed: {e}")
            self._redirect("/trades/edit")
        elif path == "/api/signals":
            with STATE_LOCK:
                out = {"rows": STATE["rows"], "last_scan": STATE["last_scan"],
                       "failed": STATE["failed"]}
            self._send(json.dumps(out, default=str), "application/json")
        elif path == "/api/options":
            with STATE_LOCK:
                out = {"options": STATE["options"],
                       "last_options": STATE["last_options"],
                       "scanning": STATE["opt_scanning"]}
            self._send(json.dumps(out, default=str), "application/json")
        elif path == "/tf":
            global EQUITY_INTERVAL, FUTURES_INTERVAL
            v = q.get("v", [""])[0]
            scope = q.get("scope", ["fut"])[0]
            if v in TF_PROFILES:
                if scope == "eq":
                    EQUITY_INTERVAL = v
                else:
                    FUTURES_INTERVAL = v
                _sync_legacy()
                print(f"[tf] {scope} -> {v} ({prof(v)['period']} of bars), rescanning")
                threading.Thread(target=run_scan, daemon=True).start()
            self._redirect(q.get("back", ["/signals"])[0])
        elif path == "/rescan":
            threading.Thread(target=run_scan, daemon=True).start()
            self._redirect("/stream")
        elif path == "/rescan-options":
            threading.Thread(target=run_options_scan, daemon=True).start()
            self._redirect("/board")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # keep the terminal clean for scan logs


def main():
    print("=" * 62)
    print("CAM'S STOCK SCANNER — STREAM SERVER")
    print("=" * 62)
    print(f"  Board:         http://localhost:{PORT}/board    "
          f"(split scanner <-> charts, {BOARD_PHASE}s flip)")
    print(f"  Combined box:  http://localhost:{PORT}/stream   (signals + chart grid)")
    print(f"  Signals box:   http://localhost:{PORT}/signals")
    print(f"  Trades box:    http://localhost:{PORT}/trades")
    print(f"  Trades edit:   http://localhost:{PORT}/trades/edit")
    print(f"  Force rescan:  http://localhost:{PORT}/rescan")
    if OPTIONS_OK and OPTIONS_ON:
        print(f"  Options JSON:  http://localhost:{PORT}/api/options")
        print(f"  Rescan opts:   http://localhost:{PORT}/rescan-options")
        print(f"  Options phase ON (board only) · every "
              f"{OPTIONS_INTERVAL // 60} min · {OPTIONS_MIN_DTE}-{OPTIONS_MAX_DTE} DTE "
              f"· delta ~{OPTIONS_TARGET_DELTA}")
    print(f"  Equities {EQUITY_INTERVAL} · futures "
          f"{FUTURES_INTERVAL if FUTURES_INTRADAY else EQUITY_INTERVAL} · "
          f"per-timeframe profiles · veto {'on' if USE_VETO else 'off'}")
    print(f"  Switch timeframe: http://localhost:{PORT}/tf?scope=fut&v=5m "
          f"(or scope=eq · {'/'.join(TF_ORDER)})")
    print(f"  Scans every {SCAN_INTERVAL // 60} min · pages refresh every {PAGE_REFRESH}s")
    print("  First scan takes ~20-30 seconds…   Ctrl+C to stop.")
    print("=" * 62)

    threading.Thread(target=scan_loop, daemon=True).start()
    if OPTIONS_OK and OPTIONS_ON:
        threading.Thread(target=options_loop, daemon=True).start()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
