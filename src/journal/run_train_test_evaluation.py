from __future__ import annotations

import pandas as pd

from .journal_config import JOURNAL_DAILY_PATH
from .journal_metrics import (
    build_summary_metrics,
    compute_drawdown_series,
    compute_equity_curve,
)

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

def evaluate_period(df: pd.DataFrame, name: str) -> None:
    daily_returns = df["net_portfolio_return"]
    turnover = df["turnover"]
    trading_cost = df["trading_cost"]

    equity_curve = compute_equity_curve(
        daily_portfolio_returns=daily_returns,
        initial_capital=1.0,
    )

    drawdown = compute_drawdown_series(equity_curve)

    benchmark = None
    if "benchmark_return" in df.columns:
        benchmark = df["benchmark_return"]

    summary = build_summary_metrics(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        drawdown_series=drawdown,
        turnover_series=turnover,
        trading_cost_series=trading_cost,
        benchmark_returns=benchmark,
    )

    print()
    print(f"{name} PERFORMANCE")
    print("-" * 40)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.6f}")
        else:
            print(f"{key:35s}: {value}")


def main() -> None:
    journal = pd.read_parquet(JOURNAL_DAILY_PATH)

    journal["date"] = pd.to_datetime(journal["date"])

    train = journal[journal["date"] <= TRAIN_END]
    test = journal[journal["date"] >= TEST_START]

    print()
    print("TRAIN / TEST SPLIT")
    print("------------------")
    print(f"Train rows: {len(train)}")
    print(f"Test rows:  {len(test)}")

    evaluate_period(train, "TRAIN (2021-2024)")
    evaluate_period(test, "TEST (2025-2026)")


if __name__ == "__main__":
    main()