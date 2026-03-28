from __future__ import annotations

from pathlib import Path

import pandas as pd

from .portfolio import PortfolioConfig, run_portfolio_engine


SIGNALS_PATH = Path("data/signals.parquet")
TARGETS_PATH = Path("data/portfolio_targets.parquet")
TRADES_PATH = Path("data/trades.parquet")
HOLDINGS_PATH = Path("data/holdings.parquet")


def main() -> None:
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {SIGNALS_PATH}")

    signals_df = pd.read_parquet(SIGNALS_PATH)
    """
    cfg = PortfolioConfig(
        top_n=10,
        rebalance_every_n_days=5,
        weight_method="inverse_vol",   # change to "equal" if you want
        max_weight=0.15,
        slippage_bps=10.0,
        min_vol_floor=1e-6,
        output_round=6,
    )
    """
    cfg = PortfolioConfig(
        top_n=10,
        rebalance_every_n_days=5,
        entry_n=15,
        exit_n=25,
        weight_method="inverse_vol",         
        max_weight=0.15,
        slippage_bps=10.0,
        min_vol_floor=1e-6,
        output_round=6,
    )

    targets_df, trades_df, holdings_df = run_portfolio_engine(
        signals_df=signals_df,
        config=cfg,
    )

    TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)

    targets_df.to_parquet(TARGETS_PATH, index=False)
    trades_df.to_parquet(TRADES_PATH, index=False)
    holdings_df.to_parquet(HOLDINGS_PATH, index=False)

    print("Portfolio targets written to:", TARGETS_PATH)
    print("Trades written to:", TRADES_PATH)
    print("Holdings written to:", HOLDINGS_PATH)

    print()
    print("Targets rows:", len(targets_df))
    print("Trades rows:", len(trades_df))
    print("Holdings rows:", len(holdings_df))

    if not targets_df.empty:
        last_reb_date = targets_df["date"].max()
        latest_targets = (
            targets_df.loc[targets_df["date"] == last_reb_date]
            .sort_values("rank")
            .reset_index(drop=True)
        )

        print()
        print("Latest rebalance date:", last_reb_date)
        print("Latest target portfolio:")
        print(latest_targets[["rank", "symbol", "composite_score", "target_weight"]].to_string(index=False))

    if not trades_df.empty:
        latest_trade_date = trades_df["rebalance_date"].max()
        latest_trades = (
            trades_df.loc[trades_df["rebalance_date"] == latest_trade_date]
            .sort_values(["side", "symbol"])
            .reset_index(drop=True)
        )

        print()
        print("Latest rebalance turnover:", latest_trades["turnover"].iloc[0])
        print("Latest trades:")
        print(
            latest_trades[
                [
                    "symbol",
                    "side",
                    "current_weight",
                    "target_weight",
                    "trade_weight",
                    "exec_date",
                    "exec_price",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()