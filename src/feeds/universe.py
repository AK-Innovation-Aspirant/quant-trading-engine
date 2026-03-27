from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


NIFTY_500_CSV_PATH = Path("data/nifty500_constituents.csv")


def _normalize_symbol_for_yfinance(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("Encountered empty symbol while building universe.")
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def get_reference_universe() -> List[str]:
    print("Loading NIFTY 500 constituent list from local CSV...")

    if not NIFTY_500_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing constituent file: {NIFTY_500_CSV_PATH}"
        )

    df = pd.read_csv(NIFTY_500_CSV_PATH)
    print("CSV loaded.")

    if "Symbol" not in df.columns:
        raise ValueError(
            f"Expected 'Symbol' column in constituent CSV, got columns: {list(df.columns)}"
        )

    symbols = sorted(
        {
            _normalize_symbol_for_yfinance(symbol)
            for symbol in df["Symbol"].dropna().tolist()
        }
    )

    if not symbols:
        raise ValueError("Reference universe is empty after loading NIFTY 500 CSV.")

    print(f"Loaded {len(symbols)} symbols.")
    return symbols