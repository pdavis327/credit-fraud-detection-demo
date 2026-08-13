"""Build Feast-ready Parquet from card_transdata.csv for manual upload to MinIO."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
FEAST_DIR = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def write_transaction_examples(df: pd.DataFrame, output_path: Path) -> None:
    fraud_ids = df.loc[df["fraud"] == 1, "transaction_id"].head(5)
    safe_ids = df.loc[df["fraud"] == 0, "transaction_id"].head(5)
    sample_ids = pd.concat([fraud_ids, safe_ids]).drop_duplicates()
    examples = [
        {
            "transaction_id": int(transaction_id),
            "label": "fraud" if df.loc[df["transaction_id"] == transaction_id, "fraud"].iloc[0] else "not fraud",
        }
        for transaction_id in sample_ids
    ]
    output_path.write_text(json.dumps(examples, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create transactions.parquet for manual upload to MinIO.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "card_transdata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEAST_DIR / "data" / "transactions.parquet",
        help="Local output file (upload this to MinIO yourself).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=10000,
        help="Rows to include (default 10000). Use --full for entire CSV.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full CSV (ignores --nrows).",
    )
    args = parser.parse_args()

    nrows = None if args.full else args.nrows
    df = pd.read_csv(args.csv, nrows=nrows)
    df.insert(0, "transaction_id", df.index.astype("int64"))
    df["event_timestamp"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="h",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    write_transaction_examples(df, FEAST_DIR / "transaction_examples.json")

    print(f"Wrote {len(df)} rows to {args.output}")
    print("Upload to MinIO:")
    print(f"  bucket: rhoai  (AWS_S3_BUCKET)")
    print(f"  key:    {os.getenv('FEAST_S3_PREFIX', 'feast/credit-fraud')}/transactions.parquet")
    print(f"Updated {FEAST_DIR / 'transaction_examples.json'}")


if __name__ == "__main__":
    main()
