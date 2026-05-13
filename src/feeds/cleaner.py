from __future__ import annotations

import pandas as pd

from .validator import validate_market_data


COLUMN_RENAME_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Symbol": "symbol",
}

MARKET_DATA_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

PRICE_TOLERANCE = 1e-8
MAX_BAD_ROW_FRACTION = 0.01


def _empty_market_data_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MARKET_DATA_COLUMNS)


def _drop_bad_market_data_rows(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    original_rows = len(df)

    if original_rows == 0:
        return df

    positive_mask = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["adj_close"] > 0)
        & (df["volume"] >= 0)
    )

    ohlc_mask = (
        (df["high"] + PRICE_TOLERANCE >= df["open"])
        & (df["high"] + PRICE_TOLERANCE >= df["close"])
        & (df["high"] + PRICE_TOLERANCE >= df["low"])
        & (df["low"] <= df["open"] + PRICE_TOLERANCE)
        & (df["low"] <= df["close"] + PRICE_TOLERANCE)
        & (df["low"] <= df["high"] + PRICE_TOLERANCE)
    )

    clean_df = df[positive_mask & ohlc_mask].copy()
    removed_rows = original_rows - len(clean_df)
    bad_row_fraction = removed_rows / original_rows

    if clean_df.empty:
        raise ValueError(f"All rows removed during market data cleaning for {symbol}.")

    if bad_row_fraction > MAX_BAD_ROW_FRACTION:
        raise ValueError(
            f"Too many bad rows removed for {symbol}: "
            f"{removed_rows}/{original_rows} ({bad_row_fraction:.2%})"
        )

    if removed_rows > 0:
        print(
            f"[WARN] Removed {removed_rows} bad market data rows for {symbol} "
            f"({bad_row_fraction:.2%})."
        )

    return clean_df


def _normalize_single_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize a single-symbol yfinance dataframe into long-format canonical schema.
    Expected raw columns from yfinance:
    Open, High, Low, Close, Adj Close, Volume
    """
    if df.empty:
        return _empty_market_data_frame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = df.reset_index().rename(columns=COLUMN_RENAME_MAP)
    out["symbol"] = symbol

    missing = [c for c in MARKET_DATA_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Downloaded dataframe for {symbol} missing columns: {missing}")

    out = out[MARKET_DATA_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=MARKET_DATA_COLUMNS)
    out["volume"] = out["volume"].astype("int64")

    out = out.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )

    out = _drop_bad_market_data_rows(out, symbol=symbol)

    return out.reset_index(drop=True)


def clean_downloaded_symbol_data(raw_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert dict[symbol -> raw yfinance df] into one clean long-format dataframe.
    """
    parts = []

    for symbol, raw_df in raw_by_symbol.items():
        cleaned = _normalize_single_symbol_df(raw_df, symbol=symbol)
        if not cleaned.empty:
            parts.append(cleaned)

    if not parts:
        return _empty_market_data_frame()

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    df = df.reset_index(drop=True)

    validate_market_data(df)
    return df