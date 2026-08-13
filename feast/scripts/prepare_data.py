"""Convert card_transdata.csv to Feast-ready Parquet with entity key and timestamp."""

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def prepare(csv_path: Path, output_path: Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    df.insert(0, "transaction_id", df.index.astype("int64"))
    df["event_timestamp"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="h",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "card_transdata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "transactions_sample.parquet",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=10000,
        help="Number of rows to include (default: 10000 for sample parquet).",
    )
    args = parser.parse_args()

    df = prepare(args.csv, args.output, nrows=args.nrows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
