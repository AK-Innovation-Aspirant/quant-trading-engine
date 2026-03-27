from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "eligible",
    "composite_score",
    "volatility_raw",
    "open",
}


@dataclass(frozen=True)
class PortfolioConfig:
    # target number of holdings
    top_n: int = 30

    # rebalance roughly monthly
    rebalance_every_n_days: int = 20

    # entry / exit rank bands to reduce churn
    entry_n: int = 150
    exit_n: int = 250

    # "equal" or "inverse_vol"
    weight_method: str = "equal"

    # lower cap to reduce concentration
    max_weight: float = 0.10
    slippage_bps: float = 10.0

    # if inverse vol is used, avoid blowups on tiny vol
    min_vol_floor: float = 1e-6

    output_round: int = 6


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"signals.parquet missing required columns: {sorted(missing)}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    numeric_cols: Iterable[str] = [
        "composite_score",
        "volatility_raw",
        "open",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "eligible" in out.columns:
        out["eligible"] = out["eligible"].fillna(False).astype(bool)

    return out


def _sorted_trading_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(df["date"].dropna().unique().tolist())


def _compute_rebalance_dates(
    trading_dates: list[pd.Timestamp],
    every_n_days: int,
) -> list[pd.Timestamp]:
    if every_n_days <= 0:
        raise ValueError("rebalance_every_n_days must be > 0")
    return trading_dates[::every_n_days]


def _cap_and_renormalize(weights: pd.Series, max_weight: float) -> pd.Series:
    """
    Iteratively cap overweight positions and redistribute the excess
    among uncapped positions proportionally to their existing weights.
    """
    if weights.empty:
        return weights

    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1].")

    w = weights.copy().astype(float)
    w = w / w.sum()

    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break

        w.loc[over] = max_weight
        remainder = 1.0 - w.loc[over].sum()
        uncapped = ~over

        if remainder < -1e-12:
            raise ValueError("max_weight too small to allocate full portfolio.")

        if uncapped.sum() == 0:
            if abs(w.sum() - 1.0) > 1e-8:
                raise ValueError("Cannot renormalize: no uncapped positions remain.")
            break

        uncapped_base = w.loc[uncapped]
        if uncapped_base.sum() <= 0:
            w.loc[uncapped] = remainder / uncapped.sum()
        else:
            w.loc[uncapped] = remainder * (uncapped_base / uncapped_base.sum())

    w = w / w.sum()
    return w


def _compute_equal_weights(symbols: list[str], max_weight: float) -> pd.Series:
    if len(symbols) == 0:
        return pd.Series(dtype=float)

    weights = pd.Series(1.0 / len(symbols), index=symbols, dtype=float)
    return _cap_and_renormalize(weights, max_weight=max_weight)


def _compute_inverse_vol_weights(
    selected: pd.DataFrame,
    max_weight: float,
    min_vol_floor: float,
) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)

    vols = selected.set_index("symbol")["volatility_raw"].copy()
    vols = vols.clip(lower=min_vol_floor)

    raw = 1.0 / vols
    raw = raw / raw.sum()

    return _cap_and_renormalize(raw, max_weight=max_weight)


def _compute_target_weights(
    selected: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.Series:
    symbols = selected["symbol"].tolist()

    if cfg.weight_method == "equal":
        return _compute_equal_weights(symbols=symbols, max_weight=cfg.max_weight)

    if cfg.weight_method == "inverse_vol":
        return _compute_inverse_vol_weights(
            selected=selected,
            max_weight=cfg.max_weight,
            min_vol_floor=cfg.min_vol_floor,
        )

    raise ValueError("weight_method must be either 'equal' or 'inverse_vol'")


def _rank_eligible_on_date(day_df: pd.DataFrame) -> pd.DataFrame:
    eligible = day_df.loc[day_df["eligible"]].copy()
    eligible = eligible.dropna(subset=["composite_score", "volatility_raw", "open"])

    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(
        ["composite_score", "symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)

    eligible["rank"] = np.arange(1, len(eligible) + 1)
    return eligible


def _previous_holdings_as_weights(
    holdings_df: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series:
    """
    Get holdings weights as of the most recent holdings snapshot strictly before `date`.
    """
    hist = holdings_df.loc[holdings_df["date"] < date].copy()
    if hist.empty:
        return pd.Series(dtype=float)

    last_date = hist["date"].max()
    snap = hist.loc[hist["date"] == last_date].copy()

    if snap.empty:
        return pd.Series(dtype=float)

    return snap.set_index("symbol")["weight"].sort_index()


def _union_weight_frame(
    old_w: pd.Series,
    new_w: pd.Series,
) -> pd.DataFrame:
    all_symbols = sorted(set(old_w.index).union(set(new_w.index)))
    frame = pd.DataFrame(index=all_symbols)
    frame["current_weight"] = old_w.reindex(all_symbols).fillna(0.0)
    frame["target_weight"] = new_w.reindex(all_symbols).fillna(0.0)
    frame["trade_weight"] = frame["target_weight"] - frame["current_weight"]
    return frame.reset_index(names="symbol")


def _compute_turnover(weights_df: pd.DataFrame) -> float:
    return 0.5 * weights_df["trade_weight"].abs().sum()


def _next_trading_open_prices(
    full_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    future = full_df.loc[full_df["date"] > rebalance_date, ["date", "symbol", "open"]].copy()
    if future.empty:
        return pd.DataFrame(columns=["symbol", "exec_date", "exec_open"])

    future = future.sort_values(["symbol", "date"])
    next_rows = future.groupby("symbol", as_index=False).first()
    next_rows = next_rows.loc[next_rows["symbol"].isin(symbols)].copy()
    next_rows = next_rows.rename(columns={"date": "exec_date", "open": "exec_open"})
    return next_rows


def _select_buffered_portfolio(
    ranked_df: pd.DataFrame,
    previous_symbols: set[str],
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    if ranked_df.empty:
        return ranked_df

    if cfg.entry_n <= 0 or cfg.exit_n <= 0 or cfg.top_n <= 0:
        raise ValueError("top_n, entry_n, and exit_n must all be > 0")

    if cfg.exit_n < cfg.entry_n:
        raise ValueError("exit_n must be >= entry_n")

    ranked = ranked_df.copy()

    # 1. Keep prior holdings if they have not deteriorated too much.
    keep_mask = ranked["symbol"].isin(previous_symbols) & (ranked["rank"] <= cfg.exit_n)
    kept = ranked.loc[keep_mask].copy()

    # 2. Add strongest new names from the entry band.
    entry_pool = ranked.loc[ranked["rank"] <= cfg.entry_n].copy()
    entry_pool = entry_pool.loc[~entry_pool["symbol"].isin(kept["symbol"])].copy()

    slots_left = max(cfg.top_n - len(kept), 0)
    adds = entry_pool.head(slots_left).copy()

    selected = pd.concat([kept, adds], ignore_index=True)

    # 3. If still underinvested, fill from the best remaining ranked names.
    if len(selected) < cfg.top_n:
        remaining = ranked.loc[~ranked["symbol"].isin(selected["symbol"])].copy()
        fill_needed = cfg.top_n - len(selected)
        fills = remaining.head(fill_needed).copy()
        selected = pd.concat([selected, fills], ignore_index=True)

    # 4. If too many names survived the keep rule, trim to best-ranked names.
    selected = selected.sort_values(["rank", "symbol"], ascending=[True, True]).head(cfg.top_n).copy()

    return selected


def _build_targets_for_date(
    full_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    previous_symbols: set[str],
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    day_df = full_df.loc[full_df["date"] == rebalance_date].copy()
    ranked = _rank_eligible_on_date(day_df)

    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "composite_score",
                "volatility_raw",
                "target_weight",
                "rank",
            ]
        )

    selected = _select_buffered_portfolio(
        ranked_df=ranked,
        previous_symbols=previous_symbols,
        cfg=cfg,
    )

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "composite_score",
                "volatility_raw",
                "target_weight",
                "rank",
            ]
        )

    target_weights = _compute_target_weights(selected=selected, cfg=cfg)
    selected["target_weight"] = selected["symbol"].map(target_weights)

    out = selected[
        [
            "date",
            "symbol",
            "composite_score",
            "volatility_raw",
            "target_weight",
            "rank",
        ]
    ].copy()

    return out


def run_portfolio_engine(
    signals_df: pd.DataFrame,
    config: PortfolioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        targets_df  : intended portfolio targets on rebalance dates
        trades_df   : simulated trades using next-day open + slippage
        holdings_df : holdings snapshots after each rebalance
    """
    cfg = config or PortfolioConfig()

    df = _normalize_columns(signals_df)
    _validate_input(df)
    df = _clean_types(df)

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    trading_dates = _sorted_trading_dates(df)
    rebalance_dates = _compute_rebalance_dates(
        trading_dates=trading_dates,
        every_n_days=cfg.rebalance_every_n_days,
    )

    all_targets: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_holdings: list[pd.DataFrame] = []

    holdings_snapshots = pd.DataFrame(columns=["date", "symbol", "weight"])

    for rebalance_date in rebalance_dates:
        old_w = _previous_holdings_as_weights(
            holdings_df=holdings_snapshots,
            date=rebalance_date,
        )
        previous_symbols = set(old_w.index.tolist())

        targets = _build_targets_for_date(
            full_df=df,
            rebalance_date=rebalance_date,
            previous_symbols=previous_symbols,
            cfg=cfg,
        )

        if targets.empty:
            continue

        new_w = targets.set_index("symbol")["target_weight"].sort_index()

        weights_df = _union_weight_frame(old_w=old_w, new_w=new_w)
        turnover = _compute_turnover(weights_df)

        exec_prices = _next_trading_open_prices(
            full_df=df,
            rebalance_date=rebalance_date,
            symbols=weights_df["symbol"].tolist(),
        )

        trade_rows = weights_df.merge(exec_prices, on="symbol", how="left")

        slip_frac = cfg.slippage_bps / 10_000.0
        trade_rows["side"] = np.where(
            trade_rows["trade_weight"] > 1e-12,
            "BUY",
            np.where(trade_rows["trade_weight"] < -1e-12, "SELL", "HOLD"),
        )

        trade_rows["exec_price"] = trade_rows["exec_open"]
        buy_mask = trade_rows["side"] == "BUY"
        sell_mask = trade_rows["side"] == "SELL"

        trade_rows.loc[buy_mask, "exec_price"] = trade_rows.loc[buy_mask, "exec_open"] * (1.0 + slip_frac)
        trade_rows.loc[sell_mask, "exec_price"] = trade_rows.loc[sell_mask, "exec_open"] * (1.0 - slip_frac)

        trade_rows["slippage_bps"] = np.where(
            trade_rows["side"] == "HOLD",
            0.0,
            cfg.slippage_bps,
        )

        trade_rows["slippage_cost_weight"] = np.where(
            trade_rows["side"] == "HOLD",
            0.0,
            trade_rows["trade_weight"].abs() * slip_frac,
        )

        trade_rows["rebalance_date"] = rebalance_date
        trade_rows["turnover"] = turnover

        trade_rows = trade_rows[
            [
                "rebalance_date",
                "symbol",
                "side",
                "current_weight",
                "target_weight",
                "trade_weight",
                "exec_date",
                "exec_open",
                "exec_price",
                "slippage_bps",
                "slippage_cost_weight",
                "turnover",
            ]
        ].copy()

        holdings = new_w.rename("weight").reset_index()
        holdings["date"] = rebalance_date
        holdings = holdings[["date", "symbol", "weight"]].copy()

        all_targets.append(targets)
        all_trades.append(trade_rows)
        all_holdings.append(holdings)

        holdings_snapshots = pd.concat(
            [holdings_snapshots, holdings],
            ignore_index=True,
        )

    targets_df = (
        pd.concat(all_targets, ignore_index=True)
        if all_targets
        else pd.DataFrame(columns=["date", "symbol", "composite_score", "volatility_raw", "target_weight", "rank"])
    )

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame(
            columns=[
                "rebalance_date",
                "symbol",
                "side",
                "current_weight",
                "target_weight",
                "trade_weight",
                "exec_date",
                "exec_open",
                "exec_price",
                "slippage_bps",
                "slippage_cost_weight",
                "turnover",
            ]
        )
    )

    holdings_df = (
        pd.concat(all_holdings, ignore_index=True)
        if all_holdings
        else pd.DataFrame(columns=["date", "symbol", "weight"])
    )

    target_round_cols = ["composite_score", "volatility_raw", "target_weight"]
    trade_round_cols = [
        "current_weight",
        "target_weight",
        "trade_weight",
        "exec_open",
        "exec_price",
        "slippage_bps",
        "slippage_cost_weight",
        "turnover",
    ]
    holding_round_cols = ["weight"]

    for col in target_round_cols:
        if col in targets_df.columns:
            targets_df[col] = targets_df[col].round(cfg.output_round)

    for col in trade_round_cols:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].round(cfg.output_round)

    for col in holding_round_cols:
        if col in holdings_df.columns:
            holdings_df[col] = holdings_df[col].round(cfg.output_round)

    return targets_df, trades_df, holdings_df