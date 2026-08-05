"""
options_scanner.py — Cam's Options Scanner

Scans option chains and prices the BET, not the hype.

For every contract it answers:
  - What % move does the stock need just to break even?
  - What's the real probability this finishes in the money?
  - How much am I losing instantly to the bid-ask spread?
  - How much value does this bleed per day (theta)?
  - Is there enough volume/OI that I can actually get out?

Requires: yfinance, pandas, numpy
    pip install yfinance pandas numpy
"""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Black-Scholes (no scipy needed — normal CDF via math.erf)
# ----------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(S, K, T, r, sigma, kind="call"):
    """
    S     = spot price
    K     = strike
    T     = years to expiration
    r     = risk-free rate (annual, decimal)
    sigma = implied volatility (annual, decimal)

    Returns dict of price + greeks + probability ITM.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(price=np.nan, delta=np.nan, gamma=np.nan,
                    theta=np.nan, vega=np.nan, prob_itm=np.nan)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if kind == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        # risk-neutral probability of finishing ITM
        prob_itm = _norm_cdf(d2)
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        prob_itm = _norm_cdf(-d2)
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))

    gamma = _norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * _norm_pdf(d1) * sqrtT / 100.0   # per 1 vol point
    theta = theta / 365.0                       # per calendar day

    return dict(price=price, delta=delta, gamma=gamma,
                theta=theta, vega=vega, prob_itm=prob_itm)


# ----------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------

class OptionsScanner:
    def __init__(self, risk_free_rate=0.043):
        self.r = risk_free_rate

    def _fetch(self, ticker, max_expirations=3):
        import yfinance as yf

        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data for {ticker}")
        spot = float(hist["Close"].iloc[-1])

        expirations = tk.options[:max_expirations]
        frames = []

        for exp in expirations:
            try:
                chain = tk.option_chain(exp)
            except Exception:
                continue
            for kind, df in (("call", chain.calls), ("put", chain.puts)):
                if df is None or df.empty:
                    continue
                d = df.copy()
                d["kind"] = kind
                d["expiration"] = exp
                frames.append(d)

        if not frames:
            raise ValueError(f"No option chains for {ticker}")

        out = pd.concat(frames, ignore_index=True)
        out["ticker"] = ticker
        out["spot"] = spot
        return out

    def _enrich(self, df):
        now = datetime.now(timezone.utc).date()
        rows = []

        for _, row in df.iterrows():
            exp_date = pd.to_datetime(row["expiration"]).date()
            dte = (exp_date - now).days
            T = max(dte, 0) / 365.0

            bid = float(row.get("bid") or 0)
            ask = float(row.get("ask") or 0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get("lastPrice") or 0)
            spread = (ask - bid) if (bid > 0 and ask > 0) else np.nan
            spread_pct = (spread / mid * 100) if (mid > 0 and not np.isnan(spread)) else np.nan

            S = float(row["spot"])
            K = float(row["strike"])
            iv = float(row.get("impliedVolatility") or 0)
            kind = row["kind"]

            g = black_scholes_greeks(S, K, T, self.r, iv, kind)

            # breakeven & required move
            if kind == "call":
                breakeven = K + mid
            else:
                breakeven = K - mid
            req_move_pct = (breakeven - S) / S * 100

            # cost to enter 1 contract at the ask
            cost = ask * 100 if ask > 0 else mid * 100

            # theta as % of position value bled per day
            theta_pct = (g["theta"] / mid * 100) if mid > 0 else np.nan

            rows.append({
                "ticker": row["ticker"],
                "kind": kind,
                "strike": K,
                "expiration": row["expiration"],
                "dte": dte,
                "spot": round(S, 2),
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 3),
                "cost_1x": round(cost, 2),
                "spread_pct": round(spread_pct, 1) if not np.isnan(spread_pct) else np.nan,
                "iv": round(iv * 100, 1),
                "delta": round(g["delta"], 3),
                "theta_day": round(g["theta"], 4),
                "theta_pct_day": round(theta_pct, 1) if not np.isnan(theta_pct) else np.nan,
                "prob_itm_pct": round(g["prob_itm"] * 100, 1),
                "breakeven": round(breakeven, 2),
                "req_move_pct": round(req_move_pct, 1),
                "volume": int(row.get("volume") or 0),
                "open_interest": int(row.get("openInterest") or 0),
            })

        return pd.DataFrame(rows)

    def scan(self, tickers, kind="call", min_dte=7, max_dte=60,
             max_spread_pct=25, min_open_interest=100, min_volume=10,
             min_prob_itm=15, max_cost=500, max_expirations=3):
        """
        Returns (passed, rejected) DataFrames.

        Defaults are deliberately strict. They will reject most lottery
        tickets. That is the point.
        """
        all_rows = []
        for t in tickers:
            try:
                raw = self._fetch(t, max_expirations=max_expirations)
                all_rows.append(self._enrich(raw))
            except Exception as e:
                print(f"  [skip] {t}: {e}")

        if not all_rows:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df = df[df["kind"] == kind]
        df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
        df = df[df["mid"] > 0]

        reasons = []
        for _, r in df.iterrows():
            why = []
            if pd.notna(r["spread_pct"]) and r["spread_pct"] > max_spread_pct:
                why.append(f"spread {r['spread_pct']}%")
            if r["open_interest"] < min_open_interest:
                why.append(f"OI {r['open_interest']}")
            if r["volume"] < min_volume:
                why.append(f"vol {r['volume']}")
            if r["prob_itm_pct"] < min_prob_itm:
                why.append(f"prob {r['prob_itm_pct']}%")
            if r["cost_1x"] > max_cost:
                why.append(f"cost ${r['cost_1x']}")
            reasons.append("; ".join(why))

        df = df.assign(reject_reason=reasons)
        passed = df[df["reject_reason"] == ""].drop(columns=["reject_reason"])
        rejected = df[df["reject_reason"] != ""]

        passed = passed.sort_values("prob_itm_pct", ascending=False)
        return passed.reset_index(drop=True), rejected.reset_index(drop=True)


# ----------------------------------------------------------------------
# Board / dashboard helper
# ----------------------------------------------------------------------

def best_contracts(tickers, signal_map=None, target_delta=0.40,
                   min_dte=21, max_dte=60, min_open_interest=100,
                   max_spread_pct=30, risk_free_rate=0.043):
    """
    For each ticker, pick the single most sensible contract to express the
    scanner's view — the one closest to target_delta that is actually liquid.

    signal_map: {"AAPL": "BUY", "TSLA": "SHORT"} -> calls for BUY, puts for SHORT.
                If omitted, everything is treated as BUY (calls).

    Returns a list of plain dicts, JSON-safe, ready for a board or a table.
    Tickers with no tradable contract are simply absent from the result.
    """
    sc = OptionsScanner(risk_free_rate)
    signal_map = signal_map or {}
    out = []

    for t in tickers:
        side = signal_map.get(t, "BUY")
        kind = "call" if side == "BUY" else "put"
        try:
            raw = sc._fetch(t, max_expirations=6)
            df = sc._enrich(raw)
        except Exception as e:
            print(f"  [options] {t}: {e}")
            continue

        df = df[df["kind"] == kind]
        df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
        df = df[(df["mid"] > 0) & (df["open_interest"] >= min_open_interest)]
        df = df[df["spread_pct"].fillna(999) <= max_spread_pct]
        if df.empty:
            continue

        df = df.assign(_gap=(df["delta"].abs() - target_delta).abs())
        row = df.sort_values("_gap").iloc[0]

        out.append({
            "ticker": t,
            "side": side,
            "kind": kind,
            "strike": float(row["strike"]),
            "expiration": str(row["expiration"]),
            "dte": int(row["dte"]),
            "spot": float(row["spot"]),
            "ask": float(row["ask"]),
            "cost": float(row["cost_1x"]),
            "delta": float(row["delta"]),
            "prob_itm": float(row["prob_itm_pct"]),
            "breakeven": float(row["breakeven"]),
            "req_move": float(row["req_move_pct"]),
            "theta_pct": float(row["theta_pct_day"]) if pd.notna(row["theta_pct_day"]) else 0.0,
            "iv": float(row["iv"]),
            "spread_pct": float(row["spread_pct"]) if pd.notna(row["spread_pct"]) else 0.0,
            "oi": int(row["open_interest"]),
        })

    out.sort(key=lambda r: -r["prob_itm"])
    return out


# ----------------------------------------------------------------------
# Single-contract sanity check — use this before ANY order
# ----------------------------------------------------------------------

def check_contract(ticker, strike, expiration, kind="call", risk_free_rate=0.043):
    """
    Point this at one specific contract someone told you to buy.
    Prints the honest numbers.
    """
    sc = OptionsScanner(risk_free_rate)
    raw = sc._fetch(ticker, max_expirations=12)
    raw = raw[(raw["strike"] == strike) &
              (raw["expiration"] == expiration) &
              (raw["kind"] == kind)]
    if raw.empty:
        print(f"Contract not found: {ticker} {strike} {kind} {expiration}")
        return None

    row = sc._enrich(raw).iloc[0]

    print(f"\n  {ticker} ${strike:g} {kind.upper()} {expiration}")
    print(f"  {'-' * 46}")
    print(f"  Stock now          ${row['spot']}")
    print(f"  Contract (ask)     ${row['ask']}  ->  ${row['cost_1x']} per contract")
    print(f"  Bid/ask spread     {row['spread_pct']}% of value lost on entry")
    print(f"  Days to expiration {row['dte']}")
    print(f"  Implied vol        {row['iv']}%")
    print()
    print(f"  Breakeven          ${row['breakeven']}")
    print(f"  Move needed        {row['req_move_pct']:+.1f}% in {row['dte']} days")
    print(f"  Prob. finish ITM   {row['prob_itm_pct']}%")
    print(f"  Prob. total loss   {100 - row['prob_itm_pct']:.1f}%")
    print(f"  Decay per day      ${abs(row['theta_day']):.4f}  ({abs(row['theta_pct_day'])}% of value/day)")
    print(f"  Liquidity          vol {row['volume']} / OI {row['open_interest']}")
    print(f"  {'-' * 46}")

    if row["prob_itm_pct"] < 10:
        print("  VERDICT: lottery ticket. Expect to lose it all.")
    elif row["prob_itm_pct"] < 25:
        print("  VERDICT: long shot. Size accordingly.")
    elif row["prob_itm_pct"] < 45:
        print("  VERDICT: real but unfavorable odds.")
    else:
        print("  VERDICT: coin-flip or better. Check the catalyst.")
    print()
    return row


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # python options_scanner.py check MD 30 2026-08-21
        check_contract(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    else:
        watchlist = ["AAPL", "AMD", "NVDA", "SOFI", "PLTR", "F", "INTC"]
        sc = OptionsScanner()
        print("Scanning...")
        passed, rejected = sc.scan(watchlist, kind="call")

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)

        print(f"\n=== PASSED ({len(passed)}) ===")
        if passed.empty:
            print("Nothing cleared the filters. That's a normal result.")
        else:
            print(passed[["ticker", "strike", "expiration", "dte", "cost_1x",
                          "prob_itm_pct", "req_move_pct", "theta_pct_day",
                          "spread_pct", "open_interest"]].head(20).to_string(index=False))

        print(f"\n=== REJECTED (showing 10 of {len(rejected)}) ===")
        if not rejected.empty:
            print(rejected[["ticker", "strike", "dte", "cost_1x",
                            "prob_itm_pct", "reject_reason"]].head(10).to_string(index=False))
