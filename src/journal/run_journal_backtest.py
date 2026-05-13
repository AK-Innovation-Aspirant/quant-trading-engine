from __future__ import annotations

import json

import pandas as pd

from .journal_config import (
    COMMISSION_BPS,
    ENABLE_REGIME_FILTER,
    ENABLE_VOLATILITY_TARGETING,
    INITIAL_CAPITAL,
    JOURNAL_DAILY_PATH,
    JOURNAL_SUMMARY_PATH,
    MAX_EXPOSURE,
    MIN_EXPOSURE,
    REGIME_EXPOSURE_WHEN_NEGATIVE,
    REGIME_EXPOSURE_WHEN_POSITIVE,
    REGIME_LOOKBACK_DAYS,
    REGIME_MIN_PERIODS,
    SLIPPAGE_BPS,
    TARGET_ANNUALIZED_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    USE_EQUAL_WEIGHT_UNIVERSE_BENCHMARK,
    VOLATILITY_LOOKBACK_DAYS,
)

from .journal_data_loader import (
    build_close_price_matrix,
    build_open_close_price_matrices,
    build_open_aware_weight_matrices,
    compute_asset_return_matrix,
    compute_daily_trading_cost_series,
    compute_daily_turnover_series,
    compute_open_aware_return_matrices,
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


def compute_realized_volatility_annualized(
    gross_returns: pd.Series,
    lookback_days: int,
    trading_days_per_year: int,
) -> pd.Series:
    realized_daily_vol = gross_returns.rolling(
        window=lookback_days,
        min_periods=lookback_days,
    ).std()

    realized_annualized_vol = realized_daily_vol * (trading_days_per_year ** 0.5)
    realized_annualized_vol.name = "realized_volatility_annualized"
    return realized_annualized_vol


def compute_volatility_target_exposure(
    gross_returns: pd.Series,
    target_annualized_volatility: float,
    lookback_days: int,
    min_exposure: float,
    max_exposure: float,
    trading_days_per_year: int,
) -> tuple[pd.Series, pd.Series]:
    realized_vol_annualized = compute_realized_volatility_annualized(
        gross_returns=gross_returns,
        lookback_days=lookback_days,
        trading_days_per_year=trading_days_per_year,
    )

    raw_exposure = target_annualized_volatility / realized_vol_annualized
    raw_exposure = raw_exposure.replace([float("inf"), -float("inf")], pd.NA)

    clipped_exposure = raw_exposure.clip(lower=min_exposure, upper=max_exposure)

    lagged_exposure = clipped_exposure.shift(1).fillna(1.0)
    lagged_exposure = lagged_exposure.clip(lower=min_exposure, upper=max_exposure)
    lagged_exposure.name = "vol_target_exposure"

    return realized_vol_annualized, lagged_exposure


def compute_regime_exposure(
    returns_for_regime_signal: pd.Series,
    lookback_days: int,
    min_periods: int,
    exposure_when_negative: float,
    exposure_when_positive: float,
) -> pd.Series:
    rolling_trailing_return = returns_for_regime_signal.rolling(
        window=lookback_days,
        min_periods=min_periods,
    ).sum()

    regime_exposure = pd.Series(
        index=returns_for_regime_signal.index,
        data=exposure_when_positive,
        dtype="float64",
    )

    regime_exposure = regime_exposure.where(
        rolling_trailing_return > 0.0,
        exposure_when_negative,
    )

    regime_exposure = regime_exposure.shift(1).fillna(exposure_when_positive)
    regime_exposure.name = "regime_exposure"

    return regime_exposure


def build_daily_journal() -> tuple[pd.DataFrame, dict]:
    market_data = load_market_data()
    holdings = load_holdings_data()
    trades = load_trades_data()

    close_matrix = build_close_price_matrix(market_data)
    asset_returns = compute_asset_return_matrix(close_matrix)

    trading_dates = asset_returns.index
    symbols = asset_returns.columns

    open_matrix, raw_close_matrix = build_open_close_price_matrices(market_data)

    overnight_returns, intraday_returns = compute_open_aware_return_matrices(
        open_matrix=open_matrix,
        close_matrix=raw_close_matrix,
    )

    overnight_returns = overnight_returns.reindex(index=trading_dates, columns=symbols)
    intraday_returns = intraday_returns.reindex(index=trading_dates, columns=symbols)

    overnight_weights, intraday_weights = build_open_aware_weight_matrices(
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

    gross_overnight_return = (overnight_weights * overnight_returns).sum(axis=1, skipna=True)
    gross_overnight_return.name = "gross_overnight_return"

    gross_intraday_return = (intraday_weights * intraday_returns).sum(axis=1, skipna=True)
    gross_intraday_return.name = "gross_intraday_return"

    gross_portfolio_return = gross_overnight_return + gross_intraday_return
    gross_portfolio_return.name = "gross_portfolio_return"

    if ENABLE_VOLATILITY_TARGETING:
        realized_vol_annualized, vol_target_exposure = compute_volatility_target_exposure(
            gross_returns=gross_portfolio_return,
            target_annualized_volatility=TARGET_ANNUALIZED_VOLATILITY,
            lookback_days=VOLATILITY_LOOKBACK_DAYS,
            min_exposure=MIN_EXPOSURE,
            max_exposure=MAX_EXPOSURE,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
    else:
        realized_vol_annualized = pd.Series(index=trading_dates, data=pd.NA, dtype="float64")
        realized_vol_annualized.name = "realized_volatility_annualized"

        vol_target_exposure = pd.Series(index=trading_dates, data=1.0, dtype="float64")
        vol_target_exposure.name = "vol_target_exposure"

    if ENABLE_REGIME_FILTER:
        regime_exposure = compute_regime_exposure(
            returns_for_regime_signal=gross_portfolio_return,
            lookback_days=REGIME_LOOKBACK_DAYS,
            min_periods=REGIME_MIN_PERIODS,
            exposure_when_negative=REGIME_EXPOSURE_WHEN_NEGATIVE,
            exposure_when_positive=REGIME_EXPOSURE_WHEN_POSITIVE,
        )
    else:
        regime_exposure = pd.Series(index=trading_dates, data=1.0, dtype="float64")
        regime_exposure.name = "regime_exposure"

    target_exposure = vol_target_exposure * regime_exposure
    target_exposure = target_exposure.clip(lower=MIN_EXPOSURE, upper=MAX_EXPOSURE)
    target_exposure.name = "target_exposure"

    scaled_gross_portfolio_return = gross_portfolio_return * target_exposure
    scaled_gross_portfolio_return.name = "scaled_gross_portfolio_return"

    net_portfolio_return = scaled_gross_portfolio_return - daily_trading_cost
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
            "gross_overnight_return": gross_overnight_return.values,
            "gross_intraday_return": gross_intraday_return.values,
            "gross_portfolio_return": gross_portfolio_return.values,
            "realized_volatility_annualized": realized_vol_annualized.values,
            "vol_target_exposure": vol_target_exposure.values,
            "regime_exposure": regime_exposure.values,
            "target_exposure": target_exposure.values,
            "scaled_gross_portfolio_return": scaled_gross_portfolio_return.values,
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