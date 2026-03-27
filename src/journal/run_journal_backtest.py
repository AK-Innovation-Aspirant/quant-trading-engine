from __future__ import annotations

import json

import pandas as pd

from .journal_config import (
    COMMISSION_BPS,
    INITIAL_CAPITAL,
    JOURNAL_DAILY_PATH,
    JOURNAL_SUMMARY_PATH,
    SLIPPAGE_BPS,
    USE_EQUAL_WEIGHT_UNIVERSE_BENCHMARK,
)
from .journal_data_loader import (
    build_close_price_matrix,
    build_daily_weight_matrix,
    compute_asset_return_matrix,
    compute_daily_trading_cost_series,
    compute_daily_turnover_series,
    load_holdings_data,
    load_market_data,
    load_trades_data,
)
from .journal_metrics import (
    build_summary_metrics,
    compute_cumulative_return_series,
    compute_drawdown_series,
    compute_equity_curve,
)


def compute_equal_weight_universe_benchmark(asset_returns: pd.DataFrame) -> pd.Series:
    benchmark_returns = asset_returns.mean(axis=1, skipna=True)
    benchmark_returns.name = "benchmark_return"
    return benchmark_returns


def build_daily_journal() -> tuple[pd.DataFrame, dict]:
    market_data = load_market_data()
    holdings = load_holdings_data()
    trades = load_trades_data()

    close_matrix = build_close_price_matrix(market_data)
    asset_returns = compute_asset_return_matrix(close_matrix)

    trading_dates = asset_returns.index
    symbols = asset_returns.columns

    daily_weights = build_daily_weight_matrix(
        holdings=holdings,
        trading_dates=trading_dates,
        symbols=symbols,
    )

    daily_turnover = compute_daily_turnover_series(
        trades=trades,
        trading_dates=trading_dates,
    )

    
    daily_trading_cost = compute_daily_trading_cost_series(
        trades=trades,
        trading_dates=trading_dates,
        slippage_bps=SLIPPAGE_BPS,
        commission_bps=COMMISSION_BPS,
    )

    gross_portfolio_return = (daily_weights * asset_returns).sum(axis=1, skipna=True)
    gross_portfolio_return.name = "gross_portfolio_return"

    net_portfolio_return = gross_portfolio_return - daily_trading_cost
    net_portfolio_return.name = "net_portfolio_return"

    equity_curve = compute_equity_curve(
        daily_portfolio_returns=net_portfolio_return,
        initial_capital=INITIAL_CAPITAL,
    )
    equity_curve.name = "portfolio_value"

    cumulative_return = compute_cumulative_return_series(equity_curve)
    cumulative_return.name = "cumulative_return"

    drawdown = compute_drawdown_series(equity_curve)
    drawdown.name = "drawdown"

    benchmark_returns = None
    if USE_EQUAL_WEIGHT_UNIVERSE_BENCHMARK:
        benchmark_returns = compute_equal_weight_universe_benchmark(asset_returns)

    journal_daily = pd.DataFrame(
        {
            "date": trading_dates,
            "gross_portfolio_return": gross_portfolio_return.values,
            "trading_cost": daily_trading_cost.values,
            "net_portfolio_return": net_portfolio_return.values,
            "turnover": daily_turnover.values,
            "portfolio_value": equity_curve.values,
            "cumulative_return": cumulative_return.values,
            "drawdown": drawdown.values,
        }
    )

    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(trading_dates).fillna(0.0)
        journal_daily["benchmark_return"] = aligned_benchmark.values
        journal_daily["excess_return"] = (
            journal_daily["net_portfolio_return"] - journal_daily["benchmark_return"]
        )

    summary = build_summary_metrics(
        daily_returns=net_portfolio_return,
        equity_curve=equity_curve,
        drawdown_series=drawdown,
        turnover_series=daily_turnover,
        trading_cost_series=daily_trading_cost,
        benchmark_returns=benchmark_returns,
    )

    return journal_daily, summary


def main() -> None:
    journal_daily, summary = build_daily_journal()

    journal_daily.to_parquet(JOURNAL_DAILY_PATH, index=False)

    with open(JOURNAL_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Journal daily output written to: {JOURNAL_DAILY_PATH}")
    print(f"Journal summary written to: {JOURNAL_SUMMARY_PATH}")
    print()

    print("Backtest summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.6f}")
        else:
            print(f"{key:35s}: {value}")


if __name__ == "__main__":
    main()