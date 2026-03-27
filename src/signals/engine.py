from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}
FUNDAMENTALS_REQUIRED_COLUMNS = {"date", "symbol", "trailing_eps"}


@dataclass(frozen=True)
class SignalConfig:
    min_history_bars: int = 126

    momentum_lookback_1: int = 63     # ~3 months
    momentum_lookback_2: int = 126    # ~6 months
    reversal_lookback: int = 5
    volatility_lookback: int = 20
    trend_sma_lookback: int = 63

    weight_momentum: float = 0.35
    weight_reversal: float = 0.15
    weight_volatility: float = 0.15
    weight_trend: float = 0.15
    weight_value: float = 0.20

    output_round: int = 6


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"market_data.parquet missing required columns: {sorted(missing)}")


def _validate_fundamentals_input(df: pd.DataFrame) -> None:
    missing = FUNDAMENTALS_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"fundamentals data missing required columns: {sorted(missing)}")


def _pick_price_column(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns:
        return "adj_close"
    if "close" in df.columns:
        return "close"
    raise ValueError("Expected either 'adj_close' or 'close' column.")


def _cross_sectional_rank_to_score(s: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Convert a cross-sectional series on a single date into a score in [-1, 1].
    ascending=True  => larger raw value gets larger score
    ascending=False => smaller raw value gets larger score
    """
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)

    if valid.sum() <= 1:
        return out

    ranked = s[valid].rank(method="average", pct=True, ascending=ascending)
    out.loc[valid] = 2.0 * ranked - 1.0
    return out


def _apply_cross_sectional_scores(
    df: pd.DataFrame,
    raw_col: str,
    out_col: str,
    ascending: bool = True,
) -> pd.DataFrame:
    df[out_col] = (
        df.groupby("date", group_keys=False)[raw_col]
        .apply(lambda s: _cross_sectional_rank_to_score(s, ascending=ascending))
    )
    return df


def _merge_fundamentals(
    market_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge sparse fundamentals snapshots onto the daily market panel.

    Current behavior:
    - merge on exact (date, symbol)
    - forward-fill trailing_eps within each symbol

    This is good enough for pipeline plumbing.
    It is not yet true point-in-time historical fundamentals handling.
    """
    df = market_df.copy()

    if fundamentals_df is None or fundamentals_df.empty:
        df["trailing_eps"] = np.nan
        return df

    f = fundamentals_df.copy()
    f.columns = [c.strip().lower() for c in f.columns]
    _validate_fundamentals_input(f)

    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str)
    f["trailing_eps"] = pd.to_numeric(f["trailing_eps"], errors="coerce")

    f = (
        f.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )

    df = df.merge(
        f[["date", "symbol", "trailing_eps"]],
        on=["date", "symbol"],
        how="left",
    )

    df["trailing_eps"] = (
        df.groupby("symbol", group_keys=False)["trailing_eps"].ffill()
    )

    return df


def compute_signals(
    market_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """
    Input:
        market_df:
            canonical long-format OHLCV dataframe
            expected columns at minimum:
            date, symbol, open, high, low, close, volume
            optional: adj_close

        fundamentals_df:
            optional fundamentals dataframe with:
            date, symbol, trailing_eps

    Output:
        one row per (date, symbol) with raw features + normalized scores + composite score
    """
    cfg = config or SignalConfig()

    df = market_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    _validate_input(df)

    price_col = _pick_price_column(df)

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Merge fundamentals before computing signals
    df = _merge_fundamentals(df, fundamentals_df)

    # Numeric safety
    numeric_cols: Iterable[str] = ["open", "high", "low", "close", "volume", "trailing_eps"]
    if price_col != "close":
        numeric_cols = list(numeric_cols) + [price_col]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("symbol", group_keys=False)

    # ------------------------------------------------------------------
    # Basic rolling state
    # ------------------------------------------------------------------
    df["ret_1d"] = g[price_col].pct_change()
    df["history_bars"] = g.cumcount() + 1

    # ------------------------------------------------------------------
    # Raw features
    # ------------------------------------------------------------------
    # 1) Medium-term momentum: average of 63d and 126d returns
    df["mom_63_raw"] = g[price_col].pct_change(cfg.momentum_lookback_1)
    df["mom_126_raw"] = g[price_col].pct_change(cfg.momentum_lookback_2)
    df["momentum_raw"] = 0.5 * df["mom_63_raw"] + 0.5 * df["mom_126_raw"]

    # 2) Short-term reversal: negative of recent 5d return
    # higher score should mean "more attractive long"
    df["reversal_raw"] = -g[price_col].pct_change(cfg.reversal_lookback)

    # 3) Volatility: lower recent realized vol is better
    df["volatility_raw"] = (
        g["ret_1d"]
        .rolling(cfg.volatility_lookback, min_periods=cfg.volatility_lookback)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252.0)
    )

    # 4) Trend: price relative to 63d SMA
    df["sma_63"] = (
        g[price_col]
        .rolling(cfg.trend_sma_lookback, min_periods=cfg.trend_sma_lookback)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["trend_raw"] = (df[price_col] / df["sma_63"]) - 1.0

    # 5) Value: earnings yield = trailing EPS / price
    # Higher earnings yield = cheaper = better
    df["earnings_yield_raw"] = df["trailing_eps"] / df[price_col]

    # Optional sanity rule:
    # negative or zero earnings are usually not a clean value signal
    df.loc[df["trailing_eps"] <= 0, "earnings_yield_raw"] = np.nan

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    required_raw_cols = [
        "momentum_raw",
        "reversal_raw",
        "volatility_raw",
        "trend_raw",
        "earnings_yield_raw",
    ]
    df["eligible"] = df["history_bars"] >= cfg.min_history_bars

    for col in required_raw_cols:
        df["eligible"] &= df[col].notna()

    # Mask ineligible rows so they never receive a score
    for col in required_raw_cols:
        df.loc[~df["eligible"], col] = np.nan

    # ------------------------------------------------------------------
    # Cross-sectional normalization to [-1, 1]
    # ------------------------------------------------------------------
    # Bigger momentum_raw is better
    df = _apply_cross_sectional_scores(df, "momentum_raw", "momentum_score", ascending=True)

    # Bigger reversal_raw is better (because we negated recent return)
    df = _apply_cross_sectional_scores(df, "reversal_raw", "reversal_score", ascending=True)

    # Smaller vol is better
    df = _apply_cross_sectional_scores(df, "volatility_raw", "volatility_score", ascending=False)

    # Bigger price / SMA deviation is better
    df = _apply_cross_sectional_scores(df, "trend_raw", "trend_score", ascending=True)

    # Bigger earnings yield is better
    df = _apply_cross_sectional_scores(df, "earnings_yield_raw", "value_score", ascending=True)

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------
    df["composite_score"] = (
        cfg.weight_momentum * df["momentum_score"]
        + cfg.weight_reversal * df["reversal_score"]
        + cfg.weight_volatility * df["volatility_score"]
        + cfg.weight_trend * df["trend_score"]
        + cfg.weight_value * df["value_score"]
    )

    # ------------------------------------------------------------------
    # Final shape
    # ------------------------------------------------------------------
    out_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        price_col,
        "volume",
        "trailing_eps",
        "history_bars",
        "eligible",
        "mom_63_raw",
        "mom_126_raw",
        "momentum_raw",
        "reversal_raw",
        "volatility_raw",
        "trend_raw",
        "earnings_yield_raw",
        "momentum_score",
        "reversal_score",
        "volatility_score",
        "trend_score",
        "value_score",
        "composite_score",
    ]

    # avoid duplicate if price_col == close
    out_cols = list(dict.fromkeys(out_cols))

    out = df[out_cols].copy()

    score_cols = [
        "trailing_eps",
        "mom_63_raw",
        "mom_126_raw",
        "momentum_raw",
        "reversal_raw",
        "volatility_raw",
        "trend_raw",
        "earnings_yield_raw",
        "momentum_score",
        "reversal_score",
        "volatility_score",
        "trend_score",
        "value_score",
        "composite_score",
    ]
    out[score_cols] = out[score_cols].round(cfg.output_round)

    return out