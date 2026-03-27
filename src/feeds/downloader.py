from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .cleaner import clean_downloaded_symbol_data
#from .nifty_50_universe import get_reference_universe
#Nifty 500
from .universe import get_reference_universe
from .validator import validate_market_data


DEFAULT_DATA_PATH = Path("data/market_data.parquet")


def _download_one_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download one symbol from yfinance.
    Use either (start/end) or period.
    """
    kwargs: dict = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
        "multi_level_index": False,
    }

    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    df = yf.download(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def download_reference_universe(
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "5y",
) -> pd.DataFrame:
    """
    Download and clean daily OHLCV data for the reference universe.
    Returns canonical long-format dataframe.
    """
    symbols = list(symbols) if symbols is not None else get_reference_universe()

    raw_by_symbol: dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Downloading {symbol}")
        try:
            raw = _download_one_symbol(symbol=symbol, start=start, end=end, period=period)
            if raw is None or raw.empty:
                print(f"[WARN] No data returned for {symbol}; skipping.")
                continue
            raw_by_symbol[symbol] = raw
        except Exception as exc:
            print(f"[WARN] Failed download for {symbol}: {exc}")

    cleaned = clean_downloaded_symbol_data(raw_by_symbol)
    return cleaned


def merge_with_existing_data(
    existing_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    merged = merged.reset_index(drop=True)
    validate_market_data(merged)
    return merged


def load_existing_market_data(parquet_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame | None:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def save_market_data(df: pd.DataFrame, parquet_path: str | Path = DEFAULT_DATA_PATH) -> None:
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    validate_market_data(df)
    df.to_parquet(parquet_path, index=False)


def update_market_data_parquet(
    parquet_path: str | Path = DEFAULT_DATA_PATH,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "5y",
) -> pd.DataFrame:
    """
    End-to-end updater:
    - download fresh data
    - merge with existing parquet
    - validate
    - save canonical dataset
    """
    existing = load_existing_market_data(parquet_path)
    new_data = download_reference_universe(
        symbols=symbols,
        start=start,
        end=end,
        period=period,
    )
    merged = merge_with_existing_data(existing, new_data)
    save_market_data(merged, parquet_path)
    return merged