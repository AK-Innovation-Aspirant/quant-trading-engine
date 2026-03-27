from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .journal_config import TRADING_DAYS_PER_YEAR


def compute_equity_curve(
    daily_portfolio_returns: pd.Series,
    initial_capital: float,
) -> pd.Series:
    growth = (1.0 + daily_portfolio_returns.fillna(0.0)).cumprod()
    equity_curve = initial_capital * growth
    return equity_curve


def compute_cumulative_return_series(equity_curve: pd.Series) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float)
    return equity_curve / equity_curve.iloc[0] - 1.0


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return drawdown


def compute_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return float("nan")

    start_value = float(equity_curve.iloc[0])
    end_value = float(equity_curve.iloc[-1])

    if start_value <= 0.0 or end_value <= 0.0:
        return float("nan")

    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if years <= 0.0:
        return float("nan")

    return (end_value / start_value) ** (1.0 / years) - 1.0


def compute_annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns.dropna()) < 2:
        return float("nan")

    daily_std = daily_returns.std(ddof=1)
    if pd.isna(daily_std):
        return float("nan")

    return float(daily_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_sharpe_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
) -> float:
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")

    daily_rf = annual_risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = clean_returns - daily_rf

    excess_std = excess_returns.std(ddof=1)
    if excess_std == 0 or pd.isna(excess_std):
        return float("nan")

    return float(excess_returns.mean() / excess_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_sortino_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
) -> float:
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")

    daily_rf = annual_risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = clean_returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0.0]

    if len(downside_returns) < 2:
        return float("nan")

    downside_std = downside_returns.std(ddof=1)
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")

    return float(excess_returns.mean() / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(drawdown_series: pd.Series) -> float:
    if drawdown_series.empty:
        return float("nan")
    return float(drawdown_series.min())


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    if pd.isna(cagr) or pd.isna(max_drawdown) or max_drawdown == 0.0:
        return float("nan")
    return float(cagr / abs(max_drawdown))


def compute_win_rate(daily_returns: pd.Series) -> float:
    clean_returns = daily_returns.dropna()
    if clean_returns.empty:
        return float("nan")
    return float((clean_returns > 0.0).mean())


def compute_profit_factor(daily_returns: pd.Series) -> float:
    clean_returns = daily_returns.dropna()
    if clean_returns.empty:
        return float("nan")

    gross_profit = clean_returns[clean_returns > 0.0].sum()
    gross_loss = -clean_returns[clean_returns < 0.0].sum()

    if gross_loss == 0.0:
        return float("nan")

    return float(gross_profit / gross_loss)


def compute_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return float("nan")
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def compute_average_turnover(turnover_series: pd.Series) -> float:
    clean_turnover = turnover_series.dropna()
    if clean_turnover.empty:
        return float("nan")
    return float(clean_turnover.mean())


def compute_annualized_turnover(turnover_series: pd.Series) -> float:
    clean_turnover = turnover_series.dropna()
    if clean_turnover.empty:
        return float("nan")
    return float(clean_turnover.mean() * TRADING_DAYS_PER_YEAR)


def compute_information_ratio(excess_return_series: pd.Series) -> float:
    clean_excess = excess_return_series.dropna()
    if len(clean_excess) < 2:
        return float("nan")

    tracking_error = clean_excess.std(ddof=1)
    if tracking_error == 0.0 or pd.isna(tracking_error):
        return float("nan")

    return float(clean_excess.mean() / tracking_error * math.sqrt(TRADING_DAYS_PER_YEAR))


def build_summary_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    drawdown_series: pd.Series,
    turnover_series: pd.Series,
    trading_cost_series: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    total_return = compute_total_return(equity_curve)
    cagr = compute_cagr(equity_curve)
    annualized_volatility = compute_annualized_volatility(daily_returns)
    sharpe_ratio = compute_sharpe_ratio(daily_returns)
    sortino_ratio = compute_sortino_ratio(daily_returns)
    max_drawdown = compute_max_drawdown(drawdown_series)
    calmar_ratio = compute_calmar_ratio(cagr, max_drawdown)
    win_rate = compute_win_rate(daily_returns)
    profit_factor = compute_profit_factor(daily_returns)
    average_daily_turnover = compute_average_turnover(turnover_series)
    annualized_turnover = compute_annualized_turnover(turnover_series)
    total_trading_cost = float(trading_cost_series.sum())

    summary = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_daily_turnover": average_daily_turnover,
        "annualized_turnover": annualized_turnover,
        "total_trading_cost": total_trading_cost,
    }

    if benchmark_returns is not None:
        aligned_strategy, aligned_benchmark = daily_returns.align(benchmark_returns, join="inner")
        excess_returns = aligned_strategy - aligned_benchmark

        summary["benchmark_total_return"] = float((1.0 + aligned_benchmark.fillna(0.0)).prod() - 1.0)
        summary["strategy_vs_benchmark_total_return"] = float((1.0 + excess_returns.fillna(0.0)).prod() - 1.0)
        summary["information_ratio"] = compute_information_ratio(excess_returns)

    return summary