"""S3/MinIO paths for Feast data sources (cluster-internal endpoint by default)."""

import os

# Defaults match MinIO in the shared-s3 namespace; override via env (shared-s3 secret).
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")
S3_ENDPOINT = os.getenv(
    "AWS_S3_ENDPOINT",
    "http://minio-service.shared-s3.svc:9000",
)
FEAST_S3_PREFIX = os.getenv("FEAST_S3_PREFIX", "feast/credit-fraud")
DEFAULT_TRANSACTIONS_PARQUET_URI = f"s3://{S3_BUCKET}/{FEAST_S3_PREFIX}/transactions.parquet"
# Override for local materialization (e.g. /tmp/transactions.parquet after oc cp).
TRANSACTIONS_PARQUET_URI = os.getenv("FEAST_TRANSACTIONS_PATH", DEFAULT_TRANSACTIONS_PARQUET_URI)
