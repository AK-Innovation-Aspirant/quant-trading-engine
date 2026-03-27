from pathlib import Path


DATA_DIR = Path("data")

MARKET_DATA_PATH = DATA_DIR / "market_data.parquet"
HOLDINGS_PATH = DATA_DIR / "holdings.parquet"
TRADES_PATH = DATA_DIR / "trades.parquet"

JOURNAL_DAILY_PATH = DATA_DIR / "journal_daily.parquet"
JOURNAL_SUMMARY_PATH = DATA_DIR / "journal_summary.json"

TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0

# Cost assumptions
SLIPPAGE_BPS = 10.0
COMMISSION_BPS = 0.0

# Optional benchmark behavior
USE_EQUAL_WEIGHT_UNIVERSE_BENCHMARK = True