from pathlib import Path

import pandas as pd

from src.signals.engine import SignalConfig, compute_signals


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    market_data_path = data_dir / "market_data.parquet"
    fundamentals_path = data_dir / "fundamentals.parquet"

    if not market_data_path.exists():
        raise FileNotFoundError(f"Could not find market data parquet file: {market_data_path}")

    if not fundamentals_path.exists():
        raise FileNotFoundError(f"Could not find fundamentals parquet file: {fundamentals_path}")

    market_df = pd.read_parquet(market_data_path)
    fundamentals_df = pd.read_parquet(fundamentals_path)

    cfg = SignalConfig(
        min_history_bars=126,
        momentum_lookback_1=63,
        momentum_lookback_2=126,
        reversal_lookback=5,
        volatility_lookback=20,
        trend_sma_lookback=63,
        weight_momentum=0.7,
        weight_reversal=0.0,
        weight_volatility=0.0,
        weight_trend=0.0,
        weight_value=0.3,
    )

    signals_df = compute_signals(
        market_df=market_df,
        fundamentals_df=fundamentals_df,
        config=cfg,
    )

    out_path = data_dir / "signals.parquet"
    signals_df.to_parquet(out_path, index=False)

    print(f"Signals written to: {out_path}")
    print(f"Rows: {len(signals_df)}")
    print(f"Symbols: {signals_df['symbol'].nunique()}")
    print(f"Date range: {signals_df['date'].min()} to {signals_df['date'].max()}")

    latest_date = signals_df["date"].max()
    latest = signals_df[
        (signals_df["date"] == latest_date) & (signals_df["eligible"])
    ].copy()
    latest = latest.sort_values("composite_score", ascending=False)

    print(f"\nLatest signal date: {latest_date}")
    print(f"Eligible symbols on latest date: {len(latest)}")

    preview_cols = [
        "date",
        "symbol",
        "trailing_eps",
        "earnings_yield_raw",
        "momentum_score",
        "reversal_score",
        "volatility_score",
        "trend_score",
        "value_score",
        "composite_score",
    ]

    print("\nTop 10 by composite score:")
    print(latest[preview_cols].head(10).to_string(index=False))

    print("\nBottom 10 by composite score:")
    print(latest[preview_cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()