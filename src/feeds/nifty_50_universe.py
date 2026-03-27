from __future__ import annotations

from typing import List


# Current-reference NIFTY 50 style universe for Stage 1.
# Keep this as a manually curated snapshot and update intentionally.
# NSE tickers on yfinance require the .NS suffix.
NIFTY50_SYMBOLS: List[str] = sorted(
    [
        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "APOLLOHOSP.NS",
        "ASIANPAINT.NS",
        "AXISBANK.NS",
        "BAJAJ-AUTO.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "BEL.NS",
        "BHARTIARTL.NS",
        "BPCL.NS",
        "BRITANNIA.NS",
        "CIPLA.NS",
        "COALINDIA.NS",
        "DRREDDY.NS",
        "EICHERMOT.NS",
        "ETERNAL.NS",      # formerly Zomato; update if needed
        "GRASIM.NS",
        "HCLTECH.NS",
        "HDFCBANK.NS",
        "HDFCLIFE.NS",
        "HEROMOTOCO.NS",
        "HINDALCO.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "INDUSINDBK.NS",
        "INFY.NS",
        "ITC.NS",
        "JIOFIN.NS",
        "JSWSTEEL.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "M&M.NS",
        "MARUTI.NS",
        "NESTLEIND.NS",
        "NTPC.NS",
        "ONGC.NS",
        "POWERGRID.NS",
        "RELIANCE.NS",
        "SBILIFE.NS",
        "SBIN.NS",
        "SHRIRAMFIN.NS",
        "SUNPHARMA.NS",
        "TATACONSUM.NS",
        "TATAMOTORS.NS",
        "TATASTEEL.NS",
        "TCS.NS",
        "TECHM.NS",
        "TRENT.NS",
        "ULTRACEMCO.NS",
    ]
)


def get_reference_universe() -> List[str]:
    """Return the reference universe as a sorted unique list."""
    symbols = sorted(set(NIFTY50_SYMBOLS))
    if not symbols:
        raise ValueError("Reference universe is empty.")
    return symbols