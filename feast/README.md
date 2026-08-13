# Feast feature repository for the credit card fraud demo.

**Definitions** (entity, feature views) live in git and are applied by the Feast operator.

**Training data** lives on MinIO (`shared-s3` namespace), not in git.

## 1. Create the Parquet file locally

```bash
pip install pandas pyarrow
python feast/scripts/prepare_data.py          # 10k rows → feast/data/transactions.parquet
python feast/scripts/prepare_data.py --full   # entire CSV
```

## 2. Upload manually to MinIO

Upload `feast/data/transactions.parquet` via the MinIO console or CLI.

| Setting | Value |
|---------|--------|
| Bucket | Same as MLflow / `shared-s3` secret (`AWS_S3_BUCKET` in `credit-card-fraud`) |
| Object key | `feast/credit-fraud/transactions.parquet` |

MinIO UI route: `https://minio-api-shared-s3.apps.<your-cluster>/`

Feast pods read this path using the in-cluster endpoint `http://minio-service.shared-s3.svc:9000` and credentials from the `shared-s3` secret.

## 3. Apply Feast definitions

Push git changes, then:

```bash
oc apply -n credit-card-fraud -f openshift/feast-instance.yaml
oc delete pod -n credit-card-fraud -l feast.dev/name=credit-fraud-feast
```

## 4. Materialize into the online store

After `feast apply`, load rows for inference (`get_online_features`):

```bash
POD=$(oc get pods -n credit-card-fraud -l feast.dev/name=credit-fraud-feast -o jsonpath='{.items[0].metadata.name}')

oc exec -n credit-card-fraud "$POD" -c online -- \
  feast materialize 2020-01-01T00:00:00 2026-12-31T00:00:00
```

The RHOAI feature-server image does not include `s3fs`. Do **not** run `pip install s3fs` without a version pin — the latest package tries to upgrade system `fsspec` and fails with permission errors.

Install a version that matches the image’s `fsspec` (2024.9.0) into a writable path, then materialize:

```bash
oc exec -n credit-card-fraud "$POD" -c online -- sh -c \
  'pip install --target /tmp/feast-pkgs s3fs==2024.9.0 && \
   PYTHONPATH=/tmp/feast-pkgs feast materialize 2020-01-01T00:00:00 2026-12-31T00:00:00'
```

This survives until the pod restarts. For a durable fix, use a custom feature-server image with `s3fs` pre-installed.

### Fallback: local Parquet (no s3fs)

```bash
oc cp feast/data/transactions.parquet "credit-card-fraud/$POD:/tmp/transactions.parquet" -c online

oc exec -n credit-card-fraud "$POD" -c online -- sh -c \
  'export FEAST_TRANSACTIONS_PATH=/tmp/transactions.parquet && feast apply && \
   feast materialize 2020-01-01T00:00:00 2026-12-31T00:00:00'
```

`feast apply` is required so the registry picks up the local path from `FEAST_TRANSACTIONS_PATH`.
