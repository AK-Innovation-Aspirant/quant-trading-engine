from src.feeds.downloader import update_market_data_parquet
from src.feeds.fundamentals_data import update_fundamentals_parquet
from src.feeds.dataset import load_market_data, get_tradable_universe_on_date


def main() -> None:
    market_df = update_market_data_parquet(
        parquet_path="data/market_data.parquet",
        period="5y",
    )

    fundamentals_df = update_fundamentals_parquet(
        parquet_path="data/fundamentals.parquet",
    )

    print("\nMARKET DATA")
    print("Downloaded rows:", len(market_df))
    print("Symbols:", market_df["symbol"].nunique())
    print("Date range:", market_df["date"].min(), "to", market_df["date"].max())

    print("\nFUNDAMENTALS DATA")
    print("Rows:", len(fundamentals_df))
    print("Symbols:", fundamentals_df["symbol"].nunique())

    if not fundamentals_df.empty:
        print("Snapshot date range:", fundamentals_df["date"].min(), "to", fundamentals_df["date"].max())

        counts = (
            fundamentals_df.groupby("symbol")["date"]
            .nunique()
            .sort_values(ascending=False)
        )

        print("Avg snapshots per symbol:", round(counts.mean(), 2))
        print("Median snapshots per symbol:", round(counts.median(), 2))

        print("\nTop symbols by snapshot count:")
        print(counts.head(10).to_string())

        print("\nSample fundamentals:")
        print(fundamentals_df.head(20).to_string(index=False))

    loaded = load_market_data("data/market_data.parquet")
    latest_date = loaded["date"].max()

    tradable = get_tradable_universe_on_date(
        as_of_date=latest_date,
        min_valid_bars=200,
        parquet_path="data/market_data.parquet",
    )

    print("\nTRADABLE UNIVERSE")
    print("Latest date:", latest_date)
    print("Tradable symbols:", len(tradable))
    print(tradable[:10])


if __name__ == "__main__":
    main()