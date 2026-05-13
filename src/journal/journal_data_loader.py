from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .journal_config import HOLDINGS_PATH, MARKET_DATA_PATH, TRADES_PATH


def _validate_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def load_market_data() -> pd.DataFrame:
    market_data = pd.read_parquet(MARKET_DATA_PATH)

    required = ["date", "symbol"]
    _validate_columns(market_data, required=required, df_name="market_data.parquet")

    if "adj_close" not in market_data.columns and "close" not in market_data.columns:
        raise ValueError("market_data.parquet must contain either 'adj_close' or 'close'.")

    market_data["date"] = pd.to_datetime(market_data["date"])
    market_data = market_data.sort_values(["date", "symbol"]).reset_index(drop=True)
    return market_data

def load_holdings_data() -> pd.DataFrame:
    holdings = pd.read_parquet(HOLDINGS_PATH)
    _validate_columns(
        holdings,
        required=["date", "symbol", "weight"],
        df_name="holdings.parquet",
    )

    holdings["date"] = pd.to_datetime(holdings["date"])
    holdings = holdings.sort_values(["date", "symbol"]).reset_index(drop=True)
    return holdings

def load_trades_data() -> pd.DataFrame:
    trades = pd.read_parquet(TRADES_PATH)

    if "date" not in trades.columns:
        if "exec_date" in trades.columns:
            trades = trades.rename(columns={"exec_date": "date"})
        elif "rebalance_date" in trades.columns:
            trades = trades.rename(columns={"rebalance_date": "date"})

    _validate_columns(
        trades,
        required=["date", "symbol"],
        df_name="trades.parquet",
    )

    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)
    return trades

def build_close_price_matrix(market_data: pd.DataFrame) -> pd.DataFrame:
    price_col = "adj_close" if "adj_close" in market_data.columns else "close"

    close_matrix = (
        market_data.pivot(index="date", columns="symbol", values=price_col)
        .sort_index()
        .astype(float)
    )
    return close_matrix

def build_open_close_price_matrices(market_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["date", "symbol", "open", "close"]
    _validate_columns(market_data, required=required, df_name="market_data.parquet")

    open_matrix = (
        market_data.pivot(index="date", columns="symbol", values="open")
        .sort_index()
        .astype(float)
    )

    close_matrix = (
        market_data.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .astype(float)
    )

    open_matrix = open_matrix.where(open_matrix > 0.0)
    close_matrix = close_matrix.where(close_matrix > 0.0)

    return open_matrix, close_matrix


def compute_open_aware_return_matrices(
    open_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_close = close_matrix.shift(1)

    overnight_returns = (open_matrix / previous_close) - 1.0
    intraday_returns = (close_matrix / open_matrix) - 1.0

    overnight_returns = overnight_returns.replace([np.inf, -np.inf], np.nan)
    intraday_returns = intraday_returns.replace([np.inf, -np.inf], np.nan)

    overnight_returns = overnight_returns.where(overnight_returns > -1.0)
    intraday_returns = intraday_returns.where(intraday_returns > -1.0)

    overnight_returns = overnight_returns.where(overnight_returns.abs() <= 5.0)
    intraday_returns = intraday_returns.where(intraday_returns.abs() <= 5.0)

    overnight_returns.name = "overnight_return"
    intraday_returns.name = "intraday_return"

    return overnight_returns, intraday_returns

def build_open_aware_weight_matrices(
    holdings: pd.DataFrame,
    trading_dates: pd.Index,
    symbols: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intraday_weight_matrix = build_daily_weight_matrix(
        holdings=holdings,
        trading_dates=trading_dates,
        symbols=symbols,
    )

    overnight_weight_matrix = intraday_weight_matrix.copy()

    return overnight_weight_matrix, intraday_weight_matrix

def compute_asset_return_matrix(close_matrix: pd.DataFrame) -> pd.DataFrame:
    prices = close_matrix.copy().astype(float)

    # invalid prices should never be used
    prices = prices.where(prices > 0.0)

    # do not forward-fill gaps before pct_change
    asset_returns = prices.pct_change(fill_method=None)

    # kill infinities
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan)

    # impossible for long cash equities
    asset_returns = asset_returns.where(asset_returns > -1.0)

    # optional but strongly recommended:
    # remove absurd spikes caused by bad data / corporate action artifacts
    asset_returns = asset_returns.where(asset_returns.abs() <= 5.0)

    return asset_returns

def build_daily_weight_matrix(
    holdings: pd.DataFrame,
    trading_dates: pd.Index,
    symbols: pd.Index,
) -> pd.DataFrame:
    # Snapshot matrix on rebalance/holdings dates only
    snapshot_matrix = (
        holdings.pivot(index="date", columns="symbol", values="weight")
        .sort_index()
        .reindex(columns=symbols)
        .fillna(0.0)
    )

    # Expand to all trading dates, carrying latest snapshot forward
    weight_matrix = (
        snapshot_matrix.reindex(trading_dates)
        .ffill()
        .fillna(0.0)
    )

    # If holdings dated at rebalance_date but become active after next open,
    # keep the 1-day lag for now.
    weight_matrix = weight_matrix.shift(1).fillna(0.0)

    return weight_matrix

def compute_daily_turnover_series(
    trades: pd.DataFrame,
    trading_dates: pd.Index,
) -> pd.Series:
    if "turnover" in trades.columns:
        turnover = (
            trades.groupby("date")["turnover"]
            .max()
            .reindex(trading_dates)
            .fillna(0.0)
        )
        return turnover.astype(float)

    if "trade_weight" in trades.columns:
        turnover = (
            trades.groupby("date")["trade_weight"]
            .apply(lambda x: float(np.abs(x).sum()))
            .reindex(trading_dates)
            .fillna(0.0)
        )
        return turnover.astype(float)

    if "weight_change" in trades.columns:
        turnover = (
            trades.groupby("date")["weight_change"]
            .apply(lambda x: float(np.abs(x).sum()))
            .reindex(trading_dates)
            .fillna(0.0)
        )
        return turnover.astype(float)

    if "target_weight" in trades.columns and "current_weight" in trades.columns:
        turnover = (
            trades.assign(
                abs_weight_change=(trades["target_weight"] - trades["current_weight"]).abs()
            )
            .groupby("date")["abs_weight_change"]
            .sum()
            .reindex(trading_dates)
            .fillna(0.0)
        )
        return turnover.astype(float)

    raise ValueError(
        "trades.parquet must contain one of: "
        "turnover, trade_weight, weight_change, or both target_weight and current_weight."
    )


def compute_daily_trading_cost_series(
    trades: pd.DataFrame,
    trading_dates: pd.Index,
    slippage_bps: float,
    commission_bps: float,
) -> pd.Series:
    if "slippage_cost_weight" in trades.columns:
        trading_cost = (
            trades.groupby("date")["slippage_cost_weight"]
            .sum()
            .reindex(trading_dates)
            .fillna(0.0)
        )

        if commission_bps != 0.0:
            turnover = compute_daily_turnover_series(trades, trading_dates)
            trading_cost = trading_cost + turnover * (commission_bps / 10_000.0)

        return trading_cost.astype(float)

    turnover = compute_daily_turnover_series(trades, trading_dates)
    total_cost_rate = (slippage_bps + commission_bps) / 10_000.0
    trading_cost = turnover * total_cost_rate
    return trading_cost.astype(float)