import json
import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import requests
import yaml
from feast import FeatureStore

URL = os.getenv("INFERENCE_ENDPOINT")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN")
FEAST_REPO_PATH = Path(os.getenv("FEAST_REPO_PATH", "/app/feast"))
FEAST_SERVING_URL = os.getenv("FEAST_SERVING_URL")

FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]
TRANSACTION_FEATURES = [f"transaction_features:{col}" for col in FEATURE_COLUMNS]
PREDICTION_THRESHOLD = 0.995


def _load_scaler_params() -> tuple[np.ndarray, np.ndarray]:
    scaler_path = FEAST_REPO_PATH / "scaler_params.json"
    with open(scaler_path) as scaler_file:
        params = json.load(scaler_file)
    if not params.get("mean") or not params.get("scale"):
        raise ValueError(
            f"Scaler params in {scaler_path} are empty. "
            "Run the training notebook to populate mean/scale before inference."
        )
    return np.array(params["mean"]), np.array(params["scale"])


def _load_transaction_choices() -> list[tuple[str, int]]:
    parquet_path = FEAST_REPO_PATH / "data" / "transactions_sample.parquet"
    df = pd.read_parquet(parquet_path)
    fraud_ids = df.loc[df["fraud"] == 1, "transaction_id"].head(5)
    safe_ids = df.loc[df["fraud"] == 0, "transaction_id"].head(5)
    sample_ids = pd.concat([fraud_ids, safe_ids]).drop_duplicates()
    choices: list[tuple[str, int]] = []
    for transaction_id in sample_ids:
        label = "fraud" if df.loc[df["transaction_id"] == transaction_id, "fraud"].iloc[0] else "not fraud"
        choices.append((f"{int(transaction_id)} (actual: {label})", int(transaction_id)))
    return choices


def _build_feast_store() -> FeatureStore:
    if FEAST_SERVING_URL:
        client_root = Path(tempfile.mkdtemp(prefix="feast_client_"))
        shutil.copytree(FEAST_REPO_PATH, client_root, dirs_exist_ok=True)
        with open(FEAST_REPO_PATH / "feature_store.yaml") as config_file:
            config = yaml.safe_load(config_file)
        config["online_store"] = {"type": "remote", "path": FEAST_SERVING_URL}
        with open(client_root / "feature_store.yaml", "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
        return FeatureStore(repo_path=str(client_root))
    return FeatureStore(repo_path=str(FEAST_REPO_PATH.resolve()))


SCALER_MEAN, SCALER_SCALE = _load_scaler_params()
FEAST_STORE = _build_feast_store()
TRANSACTION_CHOICES = _load_transaction_choices()
DEFAULT_TRANSACTION_ID = TRANSACTION_CHOICES[0][1] if TRANSACTION_CHOICES else 0


def predict(transaction_id: int) -> str:
    online_features = FEAST_STORE.get_online_features(
        features=TRANSACTION_FEATURES,
        entity_rows=[{"transaction_id": int(transaction_id)}],
    ).to_dict()

    raw_values = [
        float(online_features[f"transaction_features:{col}"][0])
        for col in FEATURE_COLUMNS
    ]
    scaled_values = (np.array(raw_values) - SCALER_MEAN) / SCALER_SCALE

    payload = {
        "inputs": [
            {
                "name": "keras_tensor",
                "shape": [1, 7],
                "datatype": "FP32",
                "data": [scaled_values.tolist()],
            },
        ]
    }
    headers = {"Content-Type": "application/json"}
    if INFERENCE_TOKEN:
        headers["Authorization"] = f"Bearer {INFERENCE_TOKEN}"

    response = requests.post(URL, json=payload, headers=headers, timeout=120)
    body = response.json()
    if "outputs" not in body:
        return f"ERROR {response.status_code}: {body}"

    prediction_score = body["outputs"][0]["data"][0]
    prediction_label = "Fraud" if prediction_score >= PREDICTION_THRESHOLD else "Not fraud"

    feature_lines = "\n".join(
        f"  {col}: {value}" for col, value in zip(FEATURE_COLUMNS, raw_values)
    )
    return (
        f"Transaction ID: {transaction_id}\n\n"
        f"Features served by Feast (raw):\n{feature_lines}\n\n"
        f"Model score: {prediction_score:.6f}\n"
        f"Prediction: {prediction_label}"
    )


with gr.Blocks(title="Predict Credit Card Fraud") as demo:
    gr.Markdown(
        "Select a transaction ID. Features are fetched from **Feast** (online store), "
        "scaled with training parameters, then sent to the ONNX model on KServe."
    )
    transaction_dropdown = gr.Dropdown(
        choices=TRANSACTION_CHOICES,
        value=DEFAULT_TRANSACTION_ID,
        label="Transaction ID",
    )
    predict_button = gr.Button("Predict")
    result_output = gr.Textbox(label="Result", lines=12)

    predict_button.click(
        fn=predict,
        inputs=transaction_dropdown,
        outputs=result_output,
    )


if __name__ == "__main__":
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
