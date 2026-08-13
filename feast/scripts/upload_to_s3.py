"""Build Parquet from CSV and upload to MinIO for Feast offline/historical features."""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FEAST_DIR = SCRIPT_DIR.parent
REPO_ROOT = FEAST_DIR.parent

FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def prepare_dataframe(csv_path: Path, nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    df.insert(0, "transaction_id", df.index.astype("int64"))
    df["event_timestamp"] = pd.date_range(
        start="2020-01-01",
        periods=len(df),
        freq="h",
    )
    return df


def write_transaction_examples(df: pd.DataFrame, output_path: Path) -> None:
    fraud_ids = df.loc[df["fraud"] == 1, "transaction_id"].head(5)
    safe_ids = df.loc[df["fraud"] == 0, "transaction_id"].head(5)
    sample_ids = pd.concat([fraud_ids, safe_ids]).drop_duplicates()
    examples = []
    for transaction_id in sample_ids:
        is_fraud = bool(df.loc[df["transaction_id"] == transaction_id, "fraud"].iloc[0])
        examples.append(
            {
                "transaction_id": int(transaction_id),
                "label": "fraud" if is_fraud else "not fraud",
            }
        )
    output_path.write_text(json.dumps(examples, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "card_transdata.csv",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=10000,
        help="Rows to include (default 10000). Omit with --full for entire CSV.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full CSV (ignores --nrows).",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("FEAST_S3_PREFIX", "feast/credit-fraud"),
    )
    args = parser.parse_args()

    nrows = None if args.full else args.nrows
    df = prepare_dataframe(args.csv, nrows=nrows)

    local_parquet = FEAST_DIR / "data" / "_upload_transactions.parquet"
    local_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_parquet, index=False)

    examples_path = FEAST_DIR / "transaction_examples.json"
    write_transaction_examples(df, examples_path)

    bucket = _require_env("AWS_S3_BUCKET")
    endpoint = _require_env("AWS_S3_ENDPOINT")
    access_key = _require_env("AWS_ACCESS_KEY_ID")
    secret_key = _require_env("AWS_SECRET_ACCESS_KEY")

    s3_key = f"{args.prefix.strip('/')}/transactions.parquet"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.upload_file(str(local_parquet), bucket, s3_key)

    local_parquet.unlink(missing_ok=True)

    print(f"Uploaded s3://{bucket}/{s3_key} ({len(df)} rows)")
    print(f"Wrote {examples_path}")


if __name__ == "__main__":
    main()
