# Quant Trading Research Engine

## Overview

This project is a **modular, end-to-end quantitative trading research engine** designed to:

- Ingest and clean market + fundamental data
- Generate interpretable cross-sectional signals
- Construct portfolios with realistic constraints
- Backtest and evaluate performance

The system is built to **validate systematic strategies before deploying real capital**, with a focus on:

- Determinism
- Interpretability
- Realistic execution modeling
- Scalable research workflows

---

## Architecture

The pipeline follows a clean, layered structure:

```
Feeds → Signals → Portfolio → Journal (Backtest + Evaluation)
```

### 1. Feeds Layer (`src/feeds`)
Handles **data ingestion, cleaning, validation, and storage**.

Responsibilities:
- Download OHLCV data (via Yahoo Finance)
- Maintain canonical dataset (`market_data.parquet`)
- Filter tradable universe (min history constraint)
- Integrate fundamentals (EPS, earnings yield)
- Apply conservative reporting lag to reduce lookahead bias in fundamentals

Key files:
- `downloader.py` — fetches raw market data
- `cleaner.py` — cleans and standardizes data
- `validator.py` — ensures schema + quality
- `fundamentals_data.py` — fetches and processes fundamentals. Fundamental data is delayed using a conservative reporting lag approximation
  to better align signal availability with public disclosures
- `dataset.py` — dataset construction logic
- `run_feeds_test.py` — CLI entry point

Output:
```
data/market_data.parquet
```

---

### 2. Signals Layer (`src/signals`)
Generates **cross-sectional alpha signals**.

Features implemented:
- Momentum (63d, 126d)
- Short-term reversal (5d)
- Volatility stability (20d)
- Trend confirmation (price vs SMA)
- Earnings yield (fundamental signal)

Processing:
- Normalize features cross-sectionally (rank-based)
- Aggregate into composite score

Key output:
```
data/signals.parquet
```

---

### 3. Portfolio Layer (`src/portfolio`)
Transforms signals into **tradable portfolios**.

Core logic:
- Select top-N stocks by score
- Weight using inverse volatility (or equal weight)
- Apply constraints:
  - Max position size
  - Volatility floor
- Rebalance periodically
- Execute trades at next-day open with slippage
- Separate overnight and intraday return attribution

Key files:
- `portfolio.py`
- `run_portfolio.py`

Outputs:
```
data/portfolio_targets.parquet
data/trades.parquet
data/holdings.parquet
```

---

### 4. Journal Layer (`src/journal`)
Handles **backtesting and evaluation**.

Metrics computed:
- Total Return
- CAGR
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Win Rate
- Turnover
- Trading Costs
- Benchmark comparison

Key files:
- `journal_metrics.py`
- `journal_data_loader.py`
- `run_journal_backtest.py`
- `run_train_test_evaluation.py`

Outputs:
```
data/journal_daily.parquet
data/journal_summary.json
```

---

## Data

### Market Data Schema

```
[date, symbol, open, high, low, close, adj_close, volume]
```

- Stored in long format
- Uses adjusted prices where available
- Supports incremental updates

### Universe

- Initially: NIFTY-50
- Extended: NIFTY-500

Constraint:
- Minimum 200 trading days required per asset

---

## How to Run

### 1. Install Dependencies

```
pip install pandas numpy yfinance pyarrow
```

---

### 2. Run Feeds Pipeline

```
python -m src.feeds.run_feeds_test
```

---

### 3. Generate Signals

```
python -m src.signals.run_signals_test
```

---

### 4. Construct Portfolio

```
python -m src.portfolio.run_portfolio
```

---

### 5. Run Backtest

```
python -m src.journal.run_journal_backtest
```

---

### 6. Train/Test Evaluation

```
python -m src.journal.run_train_test_evaluation
```

---

## Example Outputs

- ~500 stocks universe
- Multi-year backtests
- Realistic turnover and slippage modeling

Artifacts generated:

```
data/
  market_data.parquet
  signals.parquet
  portfolio_targets.parquet
  trades.parquet
  holdings.parquet
  journal_daily.parquet
  journal_summary.json
```

---

## Design Principles

### 1. Determinism
- No randomness in portfolio construction
- Fully reproducible outputs

### 2. Interpretability
- Signals are transparent and explainable
- No black-box models

### 3. Modularity
- Each layer can be swapped independently
- Easy to extend (new signals, execution models, etc.)

### 4. Realism
- Slippage modeling
- Delayed execution (next-day open)
- Explicit overnight vs intraday return attribution
- Conservative handling of fundamental reporting delays
- Turnover tracking

---

## Notebooks

Exploratory and research notebooks:

- `explore_data.ipynb`
- `evaluation.ipynb`
- `improving_strategy.ipynb`

Used for:
- Debugging
- Strategy refinement
- Visualization

---

## Limitations

- Daily data only (no intraday)
- Simplified execution model
- No transaction cost microstructure modeling
- Limited fundamental coverage
- Fundamental data uses a fixed reporting lag approximation rather than true
  point-in-time filing timestamps
- Yahoo Finance data may contain occasional historical inconsistencies or
  corporate-action anomalies requiring row-level cleaning

---

## Future Improvements

- Add intraday execution modeling
- Expand factor library
- Integrate alternative data sources
- Risk model (sector neutrality, beta control)
- Live trading integration

---

## Project Structure

```
quant_trading_engine_project/
│
├── data/
├── src/
│   ├── feeds/
│   ├── signals/
│   ├── portfolio/
│   ├── journal/
│
├── notebooks/
├── README.md
```

---

## Motivation

This project is designed as a **research-first trading system**, with the goal of:

- Building robust strategies before deployment
- Stress-testing ideas across long time horizons
- Creating a repeatable pipeline for systematic investing
