from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validator import validate_market_data


DEFAULT_DATA_PATH = Path("data/market_data.parquet")


def load_market_data(parquet_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Market data parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    validate_market_data(df)
    return df


def get_data_up_to(
    as_of_date: str | pd.Timestamp,
    parquet_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    df = load_market_data(parquet_path)
    as_of_date = pd.to_datetime(as_of_date).normalize()
    out = df[df["date"] <= as_of_date].copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def get_symbol_history_up_to(
    symbol: str,
    as_of_date: str | pd.Timestamp,
    parquet_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    df = get_data_up_to(as_of_date=as_of_date, parquet_path=parquet_path)
    out = df[df["symbol"] == symbol].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def get_tradable_universe_on_date(
    as_of_date: str | pd.Timestamp,
    min_valid_bars: int = 200,
    parquet_path: str | Path = DEFAULT_DATA_PATH,
) -> list[str]:
    """
    Active tradable universe on date t:
    symbols with at least `min_valid_bars` valid rows up to and including t.
    """
    df = get_data_up_to(as_of_date=as_of_date, parquet_path=parquet_path)
    counts = df.groupby("symbol")["date"].count()
    eligible = counts[counts >= min_valid_bars].index.tolist()
    return sorted(eligible)


def get_cross_section_on_date(
    as_of_date: str | pd.Timestamp,
    parquet_path: str | Path = DEFAULT_DATA_PATH,
    tradable_only: bool = False,
    min_valid_bars: int = 200,
) -> pd.DataFrame:
    """
    Return one row per symbol for a given date.
    Useful for signal generation over the daily cross-section.
    """
    df = load_market_data(parquet_path)
    as_of_date = pd.to_datetime(as_of_date).normalize()
    out = df[df["date"] == as_of_date].copy()

    if tradable_only:
        eligible = set(
            get_tradable_universe_on_date(
                as_of_date=as_of_date,
                min_valid_bars=min_valid_bars,
                parquet_path=parquet_path,
            )
        )
        out = out[out["symbol"].isin(eligible)].copy()

    return out.sort_values("symbol").reset_index(drop=True)