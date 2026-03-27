from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .universe import get_reference_universe


DEFAULT_FUNDAMENTALS_PATH = Path("data/fundamentals.parquet")


def _download_one_symbol_fundamentals(symbol: str) -> pd.DataFrame:
    """
    Download a minimal fundamentals snapshot for one symbol.

    v1:
    - fetch trailing EPS from yfinance
    - stamp with today's date

    Note:
    This is a current snapshot feed, not full historical point-in-time fundamentals.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception:
        return pd.DataFrame()

    trailing_eps = pd.to_numeric(info.get("trailingEps"), errors="coerce")

    if pd.isna(trailing_eps):
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp.today().normalize(),
                "symbol": symbol,
                "trailing_eps": trailing_eps,
            }
        ]
    )


def download_reference_fundamentals(
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Download trailing EPS fundamentals for the reference universe.
    Returns one row per symbol.
    """
    symbols = list(symbols) if symbols is not None else get_reference_universe()

    frames: list[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Downloading fundamentals for {symbol}")
        try:
            df = _download_one_symbol_fundamentals(symbol)
            if df.empty:
                print(f"[WARN] No fundamentals returned for {symbol}; skipping.")
                continue
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed fundamentals download for {symbol}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["trailing_eps"] = pd.to_numeric(out["trailing_eps"], errors="coerce")

    out = out.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    out = out.reset_index(drop=True)
    return out


def load_existing_fundamentals(
    parquet_path: str | Path = DEFAULT_FUNDAMENTALS_PATH,
) -> pd.DataFrame | None:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def merge_with_existing_fundamentals(
    existing_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    if merged.empty:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    merged["date"] = pd.to_datetime(merged["date"])
    merged["symbol"] = merged["symbol"].astype(str)
    merged["trailing_eps"] = pd.to_numeric(merged["trailing_eps"], errors="coerce")

    merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    merged = merged.reset_index(drop=True)
    return merged


def save_fundamentals(
    df: pd.DataFrame,
    parquet_path: str | Path = DEFAULT_FUNDAMENTALS_PATH,
) -> None:
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if out.empty:
        out = pd.DataFrame(columns=["date", "symbol", "trailing_eps"])
    else:
        out["date"] = pd.to_datetime(out["date"])
        out["symbol"] = out["symbol"].astype(str)
        out["trailing_eps"] = pd.to_numeric(out["trailing_eps"], errors="coerce")

    out.to_parquet(parquet_path, index=False)


def update_fundamentals_parquet(
    parquet_path: str | Path = DEFAULT_FUNDAMENTALS_PATH,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    End-to-end updater:
    - download fresh trailing EPS snapshots
    - merge with existing parquet
    - save fundamentals dataset
    """
    existing = load_existing_fundamentals(parquet_path)
    new_data = download_reference_fundamentals(symbols=symbols)
    merged = merge_with_existing_fundamentals(existing, new_data)
    save_fundamentals(merged, parquet_path)
    return merged