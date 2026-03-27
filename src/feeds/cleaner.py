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


def _normalize_single_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize a single-symbol yfinance dataframe into long-format canonical schema.
    Expected raw columns from yfinance:
    Open, High, Low, Close, Adj Close, Volume
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = df.reset_index().rename(columns=COLUMN_RENAME_MAP)
    out["symbol"] = symbol

    required = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Downloaded dataframe for {symbol} missing columns: {missing}")

    out = out[required].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    out["volume"] = out["volume"].astype("int64")

    out = out.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
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
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    df = df.reset_index(drop=True)

    validate_market_data(df)
    return df