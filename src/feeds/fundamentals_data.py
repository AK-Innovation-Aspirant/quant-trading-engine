from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .universe import get_reference_universe


DEFAULT_FUNDAMENTALS_PATH = Path("data/fundamentals.parquet")


def _extract_quarterly_net_income(financials: pd.DataFrame) -> pd.Series:
    if financials is None or financials.empty:
        return pd.Series(dtype=float)

    candidates = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
        "NetIncome",
    ]

    for name in candidates:
        if name in financials.index:
            s = financials.loc[name]
            s.index = pd.to_datetime(s.index)
            s = pd.to_numeric(s, errors="coerce").sort_index()
            return s

    return pd.Series(dtype=float)


def _extract_quarterly_diluted_shares(financials: pd.DataFrame) -> pd.Series:
    if financials is None or financials.empty:
        return pd.Series(dtype=float)

    candidates = [
        "Diluted Average Shares",
        "Diluted Weighted Average Shares",
        "Weighted Average Dilution Earnings Shares",
        "Ordinary Shares Number",
        "Share Issued",
    ]

    for name in candidates:
        if name in financials.index:
            s = financials.loc[name]
            s.index = pd.to_datetime(s.index)
            s = pd.to_numeric(s, errors="coerce").sort_index()
            return s

    return pd.Series(dtype=float)


def _extract_historical_ttm_eps(symbol: str) -> pd.DataFrame:
    """
    Build approximate historical TTM EPS snapshots for one symbol.

    Output columns:
    - date
    - symbol
    - trailing_eps
    """
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    quarterly_financials = pd.DataFrame()
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            candidate = getattr(ticker, attr)
            if candidate is not None and not candidate.empty:
                quarterly_financials = candidate
                break
        except Exception:
            continue

    if quarterly_financials is None or quarterly_financials.empty:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    net_income = _extract_quarterly_net_income(quarterly_financials)
    diluted_shares = _extract_quarterly_diluted_shares(quarterly_financials)

    if net_income.empty or diluted_shares.empty:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    df = pd.DataFrame(
        {
            "net_income": net_income,
            "diluted_shares": diluted_shares,
        }
    ).sort_index()

    df["quarterly_eps"] = df["net_income"] / df["diluted_shares"]
    df["quarterly_eps"] = pd.to_numeric(df["quarterly_eps"], errors="coerce")
    df = df.dropna(subset=["quarterly_eps"])

    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    df["trailing_eps"] = df["quarterly_eps"].rolling(window=4, min_periods=4).sum()
    df = df.dropna(subset=["trailing_eps"])

    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "trailing_eps"])

    out = df.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = symbol
    out["trailing_eps"] = pd.to_numeric(out["trailing_eps"], errors="coerce")

    out = (
        out[["date", "symbol", "trailing_eps"]]
        .dropna(subset=["date", "symbol", "trailing_eps"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    return out


def _download_one_symbol_fundamentals(symbol: str) -> pd.DataFrame:
    """
    Download approximate historical trailing EPS snapshots for one symbol.
    Returns many rows across time, not one current snapshot row.
    """
    return _extract_historical_ttm_eps(symbol)


def download_reference_fundamentals(
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Download approximate historical trailing EPS snapshots
    for the reference universe.
    """
    symbols = list(symbols) if symbols is not None else get_reference_universe()

    frames: list[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Downloading fundamentals for {symbol}")
        try:
            df = _download_one_symbol_fundamentals(symbol)
            if df.empty:
                print(f"[WARN] No historical fundamentals returned for {symbol}; skipping.")
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

    out = (
        out.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )
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

    merged = (
        merged.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )
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
    - download historical trailing EPS snapshots
    - merge with existing parquet
    - save fundamentals dataset
    """
    existing = load_existing_fundamentals(parquet_path)
    new_data = download_reference_fundamentals(symbols=symbols)
    merged = merge_with_existing_fundamentals(existing, new_data)
    save_fundamentals(merged, parquet_path)
    return merged