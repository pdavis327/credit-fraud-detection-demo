# Feast feature repository for the credit card fraud demo.

**Definitions** (entity, feature views) live in git and are applied by the Feast operator.

**Training data** lives in MinIO (`shared-s3` namespace), not in this repo.

## Upload data to MinIO

From a machine with access to the cluster secret values (workbench or laptop with `oc`):

```bash
# Load credentials from the credit-card-fraud namespace secret
export AWS_ACCESS_KEY_ID=$(oc get secret shared-s3 -n credit-card-fraud -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
export AWS_SECRET_ACCESS_KEY=$(oc get secret shared-s3 -n credit-card-fraud -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
export AWS_S3_BUCKET=$(oc get secret shared-s3 -n credit-card-fraud -o jsonpath='{.data.AWS_S3_BUCKET}' | base64 -d)
export AWS_S3_ENDPOINT=$(oc get secret shared-s3 -n credit-card-fraud -o jsonpath='{.data.AWS_S3_ENDPOINT}' | base64 -d)

pip install boto3 pandas pyarrow
python feast/scripts/upload_to_s3.py          # 10k rows (default)
python feast/scripts/upload_to_s3.py --full   # entire CSV
```

Object path: `s3://<AWS_S3_BUCKET>/feast/credit-fraud/transactions.parquet`

**In-cluster endpoint** (used by Feast pods): `http://minio-service.shared-s3.svc:9000`  
**External route** (console / laptop): `https://minio-api-shared-s3.apps.<cluster>/`

After uploading, re-apply the FeatureStore CR if you changed git definitions, then restart the feast pod or run `feast apply` in the online container.
