"""
scanner.py

Pulls price data via yfinance for the configured watchlist, computes
indicators, and flags tickers matching simple scan conditions
(e.g. RSI oversold/overbought, SMA crossovers).

Data is downloaded in batches rather than one request per ticker, which
keeps a 100+ ticker watchlist to a handful of requests instead of 100+.
"""

import yfinance as yf
import pandas as pd

import config
import indicators

# How many symbols to request per yfinance call. 20 is a safe middle ground:
# big enough to cut request count ~20x, small enough that one bad symbol
# doesn't poison a huge chunk of the scan.
CHUNK_SIZE = getattr(config, "DOWNLOAD_CHUNK_SIZE", 20)


def _split_frame(raw: pd.DataFrame, chunk: list) -> dict:
    """
    yfinance returns different column shapes depending on how many symbols
    were requested. Normalize into {ticker: single-ticker OHLCV DataFrame}.
    """
    frames = {}

    if raw is None or raw.empty:
        return frames

    cols = raw.columns

    # Single symbol requested -> flat columns, no ticker level.
    if not isinstance(cols, pd.MultiIndex):
        df = raw.dropna(how="all")
        if not df.empty:
            frames[chunk[0]] = df
        return frames

    level0 = set(cols.get_level_values(0))
    level1 = set(cols.get_level_values(1))

    for ticker in chunk:
        try:
            if ticker in level0:          # group_by="ticker"
                df = raw[ticker]
            elif ticker in level1:        # group_by="column"
                df = raw.xs(ticker, axis=1, level=1)
            else:
                continue
        except (KeyError, IndexError):
            continue

        df = df.dropna(how="all")
        if not df.empty:
            frames[ticker] = df

    return frames


def fetch_history_batch(tickers, period=None, interval=None, chunk_size=CHUNK_SIZE) -> dict:
    """
    Download OHLCV history for many tickers in chunked batch requests.

    Returns {ticker: DataFrame}. Symbols that returned nothing are simply
    absent from the dict — callers should treat a missing key as "no data".
    """
    period = period or config.HISTORY_PERIOD
    interval = interval or config.HISTORY_INTERVAL

    tickers = [t for t in dict.fromkeys(tickers) if t]  # dedupe, keep order
    frames = {}

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            print(f"Batch download failed for {chunk[0]}...{chunk[-1]}: {e}")
            continue

        frames.update(_split_frame(raw, chunk))

    return frames


def fetch_history(ticker: str) -> pd.DataFrame:
    """Fetch historical OHLCV data for a single ticker (kept for compatibility)."""
    frames = fetch_history_batch([ticker])
    if ticker not in frames:
        raise ValueError(f"No data returned for {ticker}")
    return frames[ticker]


def evaluate_ticker(ticker: str, df: pd.DataFrame = None) -> dict:
    """
    Compute indicators and return a summary dict with the latest values
    and any signal flags. Pass `df` to reuse data from a batch download;
    omit it and this fetches the ticker on its own.
    """
    if df is None:
        df = fetch_history(ticker)

    df = indicators.add_all_indicators(df, config)
    latest = df.iloc[-1]

    signals = []
    if pd.notna(latest["RSI"]):
        if latest["RSI"] >= config.RSI_OVERBOUGHT:
            signals.append("RSI_OVERBOUGHT")
        if latest["RSI"] <= config.RSI_OVERSOLD:
            signals.append("RSI_OVERSOLD")

    if pd.notna(latest["SMA_short"]) and pd.notna(latest["SMA_long"]):
        if latest["SMA_short"] > latest["SMA_long"]:
            signals.append("SMA_BULLISH_CROSS")
        elif latest["SMA_short"] < latest["SMA_long"]:
            signals.append("SMA_BEARISH_CROSS")

    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        if latest["MACD"] > latest["MACD_signal"]:
            signals.append("MACD_BULLISH")
        elif latest["MACD"] < latest["MACD_signal"]:
            signals.append("MACD_BEARISH")

    if pd.notna(latest["BB_upper"]) and latest["Close"] >= latest["BB_upper"]:
        signals.append("BB_UPPER_BREAKOUT")
    if pd.notna(latest["BB_lower"]) and latest["Close"] <= latest["BB_lower"]:
        signals.append("BB_LOWER_BREAKOUT")

    def _r(key, digits=2):
        return round(float(latest[key]), digits) if pd.notna(latest[key]) else None

    return {
        "ticker": ticker,
        "close": _r("Close"),
        "rsi": _r("RSI"),
        "sma_short": _r("SMA_short"),
        "sma_long": _r("SMA_long"),
        "ema": _r("EMA"),
        "macd": _r("MACD", 4),
        "macd_signal": _r("MACD_signal", 4),
        "bb_upper": _r("BB_upper"),
        "bb_lower": _r("BB_lower"),
        "signals": signals,
    }


def run_scan(watchlist=None) -> list:
    """Run the scan across the watchlist and return a list of result dicts."""
    watchlist = watchlist or config.WATCHLIST
    frames = fetch_history_batch(watchlist)

    results = []
    missing = []

    for ticker in watchlist:
        df = frames.get(ticker)
        if df is None:
            missing.append(ticker)
            continue
        try:
            results.append(evaluate_ticker(ticker, df))
        except Exception as e:
            print(f"Error evaluating {ticker}: {e}")

    if missing:
        print(f"No data for {len(missing)} symbol(s): {', '.join(missing)}")

    return results


if __name__ == "__main__":
    scan_results = run_scan()
    for r in scan_results:
        flag_str = ", ".join(r["signals"]) if r["signals"] else "no signals"
        close = r["close"] if r["close"] is not None else "n/a"
        rsi_val = r["rsi"] if r["rsi"] is not None else "n/a"
        print(f"{r['ticker']:8} | Close: {close:>9} | RSI: {rsi_val:>6} | {flag_str}")
