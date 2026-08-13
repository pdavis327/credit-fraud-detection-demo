from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
)

transaction_source = FileSource(
    name="transaction_source",
    path="data/transactions_sample.parquet",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="distance_from_home", dtype=Float64),
        Field(name="distance_from_last_transaction", dtype=Float64),
        Field(name="ratio_to_median_purchase_price", dtype=Float64),
        Field(name="repeat_retailer", dtype=Float64),
        Field(name="used_chip", dtype=Float64),
        Field(name="used_pin_number", dtype=Float64),
        Field(name="online_order", dtype=Float64),
    ],
    source=transaction_source,
)
