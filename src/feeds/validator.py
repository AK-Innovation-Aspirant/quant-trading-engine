from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_duplicate_symbol_dates(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(subset=["date", "symbol"], keep=False)
    if duplicated.any():
        bad_rows = df.loc[duplicated, ["date", "symbol"]].head(10)
        raise ValueError(
            "Duplicate (date, symbol) rows found. Examples:\n"
            f"{bad_rows.to_string(index=False)}"
        )


def validate_price_volume_sanity(df: pd.DataFrame) -> None:
    price_cols = ["open", "high", "low", "close", "adj_close"]

    for col in price_cols:
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values.")

    if (df["volume"] < 0).any():
        raise ValueError("Column 'volume' contains negative values.")

    high_invalid = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    )
    if high_invalid.any():
        raise ValueError("Some rows violate high >= open/close/low.")

    low_invalid = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )
    if low_invalid.any():
        raise ValueError("Some rows violate low <= open/close/high.")


def validate_sorted(df: pd.DataFrame) -> None:
    expected = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    actual = df.reset_index(drop=True)
    if not actual[["symbol", "date"]].equals(expected[["symbol", "date"]]):
        raise ValueError("Data is not sorted by ['symbol', 'date'].")


def validate_market_data(df: pd.DataFrame) -> None:
    validate_required_columns(df)
    validate_no_duplicate_symbol_dates(df)
    validate_price_volume_sanity(df)
    validate_sorted(df)


def count_valid_bars_by_symbol(df: pd.DataFrame) -> pd.Series:
    validate_required_columns(df)
    return df.groupby("symbol")["date"].count().sort_index()


def symbols_with_min_history(
    df: pd.DataFrame,
    min_valid_bars: int,
) -> list[str]:
    counts = count_valid_bars_by_symbol(df)
    return counts[counts >= min_valid_bars].index.tolist()