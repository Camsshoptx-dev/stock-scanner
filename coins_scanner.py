"""
coins_scanner.py — Solana token risk check (DexScreener)

Built for the Axiom universe: brand-new Solana pairs, not CoinGecko listings.

WHAT THIS IS FOR
    Checking a token before you buy it. Paste a mint address or symbol,
    get liquidity, age, buy/sell skew, and — the number that actually
    matters — how far YOUR order size moves the price.

WHAT THIS IS NOT FOR
    Finding tokens first. A REST poll cannot beat Axiom's own infra to an
    entry, and anything that shows up here with clean numbers has already
    been seen. Use it as the check between "Axiom showed me this" and
    "I clicked buy."

WHAT IT CANNOT SEE
    DexScreener exposes market data only. It does NOT tell you whether the
    LP is burned or locked, whether mint authority was revoked, whether
    freeze authority is live, or how much supply the deployer holds. Those
    are the actual rug vectors and they need RugCheck or a Solana RPC call.
    A token can look clean on every metric below and still be a honeypot.

Requires: requests, pandas, numpy
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

API = "https://api.dexscreener.com"
UA = {"User-Agent": "cams-scanner/1.0"}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _num(v, default=np.nan):
    """NaN-safe coercion. DexScreener omits fields on brand-new pairs."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if (np.isnan(f) or np.isinf(f)) else f


def _dig(d, *keys, default=None):
    """Safe nested lookup: _dig(pair, 'liquidity', 'usd')"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _get(path, params=None, retries=3):
    import requests

    for attempt in range(retries):
        try:
            r = requests.get(f"{API}{path}", params=params or {},
                             headers=UA, timeout=20)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
            continue

        if r.status_code == 429:
            if attempt == retries - 1:
                raise RuntimeError("DexScreener rate limit (300/min). Slow down.")
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"DexScreener {r.status_code}: {r.text[:160]}")
        return r.json()

    raise RuntimeError("DexScreener unreachable")


# ----------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------

def lookup(query, chain="solana"):
    """Search by mint address, symbol, or name. Returns raw pair dicts.

    Paste the mint address straight from Axiom for an exact match.
    A bare symbol returns every impostor using that ticker, which is
    itself worth seeing.
    """
    data = _get("/latest/dex/search", {"q": query})
    pairs = data.get("pairs") or []
    if chain:
        pairs = [p for p in pairs if p.get("chainId") == chain]
    return pairs


def token_pairs(mint, chain="solana"):
    """All pairs for one mint address."""
    data = _get(f"/latest/dex/tokens/{mint}")
    pairs = data.get("pairs") or data.get("pair") or []
    if isinstance(pairs, dict):
        pairs = [pairs]
    if chain:
        pairs = [p for p in pairs if p.get("chainId") == chain]
    return pairs


# ----------------------------------------------------------------------
# the number that actually matters
# ----------------------------------------------------------------------

def price_impact(trade_usd, liquidity_usd):
    """Rough price impact of a trade against a constant-product pool.

    DexScreener reports TOTAL pool liquidity (both sides), so one side is
    about half. For xy=k, swapping dx against reserve x moves price by
    roughly dx/(x+dx).

    Approximation. Concentrated liquidity, multiple pools, and routing all
    change it. Treat it as a floor on the damage, not a ceiling — and
    remember you pay it TWICE, once in and once out.
    """
    if np.isnan(liquidity_usd) or liquidity_usd <= 0 or trade_usd <= 0:
        return np.nan
    side = liquidity_usd / 2.0
    return float(trade_usd / (side + trade_usd) * 100)


def survivable_size(liquidity_usd, max_impact_pct=2.0):
    """Largest trade that stays under max_impact_pct of slippage."""
    if np.isnan(liquidity_usd) or liquidity_usd <= 0:
        return np.nan
    side = liquidity_usd / 2.0
    m = max_impact_pct / 100.0
    return float(side * m / (1 - m))


# ----------------------------------------------------------------------
# enrich
# ----------------------------------------------------------------------

def enrich(pairs, trade_size_usd=250.0):
    rows = []
    now = datetime.now(timezone.utc)

    for p in pairs:
        liq = _num(_dig(p, "liquidity", "usd"))
        fdv = _num(p.get("fdv"))
        mcap = _num(p.get("marketCap"))
        v24 = _num(_dig(p, "volume", "h24"))
        v1 = _num(_dig(p, "volume", "h1"))

        created = p.get("pairCreatedAt")
        if created:
            try:
                age_h = (now - datetime.fromtimestamp(created / 1000, timezone.utc)
                         ).total_seconds() / 3600
            except (ValueError, OSError, OverflowError):
                age_h = np.nan
        else:
            age_h = np.nan

        b24 = _num(_dig(p, "txns", "h24", "buys"), 0)
        s24 = _num(_dig(p, "txns", "h24", "sells"), 0)
        b1 = _num(_dig(p, "txns", "h1", "buys"), 0)
        s1 = _num(_dig(p, "txns", "h1", "sells"), 0)
        tot24 = b24 + s24
        buy_share_24 = (b24 / tot24 * 100) if tot24 > 0 else np.nan
        buy_share_1 = (b1 / (b1 + s1) * 100) if (b1 + s1) > 0 else np.nan

        rows.append({
            "symbol": str(_dig(p, "baseToken", "symbol") or "?").upper(),
            "name": _dig(p, "baseToken", "name") or "",
            "mint": _dig(p, "baseToken", "address") or "",
            "dex": p.get("dexId") or "",
            "url": p.get("url") or "",
            "price_usd": _num(p.get("priceUsd")),
            "liquidity": liq,
            "fdv": fdv,
            "mcap": mcap,
            "vol_24h": v24,
            "vol_1h": v1,
            "age_h": age_h,
            "chg_5m": _num(_dig(p, "priceChange", "m5")),
            "chg_1h": _num(_dig(p, "priceChange", "h1")),
            "chg_24h": _num(_dig(p, "priceChange", "h24")),
            "buys_24h": int(b24), "sells_24h": int(s24),
            "buy_share_24h": buy_share_24,
            "buy_share_1h": buy_share_1,
            "impact_pct": price_impact(trade_size_usd, liq),
            "max_size_2pct": survivable_size(liq, 2.0),
            "trade_size": trade_size_usd,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.assign(**risk_columns(df))


def risk_columns(df):
    """Facts about exit difficulty. Not predictions about direction."""
    scores, notes = [], []

    for _, r in df.iterrows():
        s, n = 0, []
        liq, age, v24 = r["liquidity"], r["age_h"], r["vol_24h"]

        # --- liquidity: the single most important number
        if np.isnan(liq):
            s += 4; n.append("no liquidity data")
        elif liq < 5_000:
            s += 5; n.append(f"${liq:,.0f} liquidity — you cannot exit")
        elif liq < 20_000:
            s += 3; n.append(f"${liq:,.0f} liquidity — very thin")
        elif liq < 100_000:
            s += 1; n.append(f"${liq/1000:.0f}k liquidity")

        # --- your own slippage
        imp = r["impact_pct"]
        if not np.isnan(imp):
            if imp > 10:
                s += 3; n.append(f"${r['trade_size']:.0f} order moves price {imp:.0f}%")
            elif imp > 3:
                s += 1; n.append(f"{imp:.1f}% slippage on your size")

        # --- age
        if not np.isnan(age):
            if age < 1:
                s += 3; n.append(f"{age*60:.0f} minutes old")
            elif age < 24:
                s += 2; n.append(f"{age:.0f}h old")
            elif age < 72:
                s += 1; n.append(f"{age/24:.0f}d old")

        # --- churn: volume far above the pool means constant round-tripping
        if not (np.isnan(v24) or np.isnan(liq)) and liq > 0:
            churn = v24 / liq
            if churn > 20:
                s += 2; n.append(f"volume {churn:.0f}x liquidity — heavy churn")
            elif churn < 0.1:
                s += 2; n.append("almost no volume")

        # --- distribution: sells dominating
        bs = r["buy_share_1h"]
        if not np.isnan(bs):
            if bs < 35:
                s += 2; n.append(f"only {bs:.0f}% buys last hour — distribution")
            elif bs > 85:
                s += 1; n.append(f"{bs:.0f}% buys — one-sided, no exit liquidity yet")

        # --- float vs pool: big notional on a tiny pool
        if not (np.isnan(r["fdv"]) or np.isnan(liq)) and liq > 0:
            ratio = r["fdv"] / liq
            if ratio > 100:
                s += 2; n.append(f"FDV {ratio:.0f}x the pool")

        # --- already vertical
        c1 = r["chg_1h"]
        if not np.isnan(c1) and c1 > 100:
            s += 1; n.append(f"+{c1:.0f}% in the last hour")

        scores.append(s)
        notes.append("; ".join(n) if n else "no flags raised")

    return {"risk": scores, "risk_notes": notes}


def check(query, trade_size_usd=250.0, chain="solana"):
    """One-shot lookup, ranked by ACTUAL ACTIVITY.

    Sorting by liquidity is misleading: a large dormant pool outranks the
    live one. A deep pool nobody trades is not an exit, it is a number on
    a screen. Rank by 24h volume, tie-break on liquidity.
    """
    pairs = lookup(query, chain=chain)
    if not pairs:
        return pd.DataFrame()
    df = enrich(pairs, trade_size_usd=trade_size_usd)
    if df.empty:
        return df
    return (df.sort_values(["vol_24h", "liquidity"], ascending=False)
              .reset_index(drop=True))


def short_mint(m, n=4):
    """DezXAZ8z7Pn...pPB263 — enough to compare against Axiom."""
    if not m or len(m) <= n * 2 + 3:
        return m or "—"
    return f"{m[:n]}...{m[-n:]}"


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "bonk"
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0

    df = check(q, trade_size_usd=size)
    if df.empty:
        print(f"nothing found for {q!r}")
        raise SystemExit

    df = df.assign(mint_short=df["mint"].map(short_mint))

    pd.set_option("display.width", 240)
    print(f"\n{len(df)} pair(s) matching {q!r}   |   order size ${size:,.0f}")
    print("ranked by 24h volume — a deep pool with no volume is not an exit\n")

    show = df.head(10).copy()
    show["liquidity"] = show["liquidity"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    show["vol_24h"] = show["vol_24h"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    show["age"] = show["age_h"].map(
        lambda h: "—" if pd.isna(h) else
        (f"{h*60:.0f}m" if h < 1 else (f"{h:.0f}h" if h < 48 else f"{h/24:.0f}d")))
    show["impact"] = show["impact_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}%")
    show["max_2%"] = show["max_size_2pct"].map(lambda v: "—" if pd.isna(v) else f"${v:,.0f}")

    print(show[["mint_short", "dex", "liquidity", "vol_24h", "age",
                "impact", "max_2%", "risk"]].to_string(index=False))

    print()
    for _, r in df.head(5).iterrows():
        print(f"  {r['mint_short']:14} risk {r['risk']:2}  {r['risk_notes']}")

    live = df[df["vol_24h"].fillna(0) > 1000]
    dead = len(df) - len(live)
    print()
    if dead:
        print(f"  {dead} of {len(df)} pairs have under $1k daily volume — dormant pools.")
    if len(df) > 1:
        print(f"  {len(df)} pairs share this ticker. Copy the mint address out of "
              f"Axiom and pass THAT, not the symbol — impostors are routine.")
