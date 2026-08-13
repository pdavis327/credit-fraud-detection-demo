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
oc delete pod -n credit-card-fraud -l feast.dev/feature-store=credit-fraud-feast
```
