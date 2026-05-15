# Import the dependencies we need to run the code.
import os
import requests
import json
import gradio as gr
import numpy as np

# Get a few environment variables. These are so we:
# - Know what endpoint we should request
# - Set server name and port for Gradio
URL = os.getenv("INFERENCE_ENDPOINT")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
# Optional: OpenShift / model route often requires Authorization: Bearer <token>
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN")

# Create a small function that sends data to the inference endpoint and recieves a prediction
def predict(distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    payload = {
        "inputs": [
            {
                "name": "keras_tensor", 
                "shape": [1, 7], 
                "datatype": "FP32",
                "data": [[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]]
            },
            ]
        }
    headers = {"Content-Type": "application/json"}
    if INFERENCE_TOKEN:
        headers["Authorization"] = f"Bearer {INFERENCE_TOKEN}"

    response = requests.post(URL, json=payload, headers=headers, timeout=120)
    body = response.json()
    if 'outputs' not in body:
        return f"ERROR {response.status_code}: {body}"
    prediction = body['outputs'][0]['data'][0]

    return "Fraud" if prediction >= 0.995 else "Not fraud"


# Create and launch a Gradio interface that uses the prediction function to predict an output based on the inputs. 
# We also set up a few example inputs to make it easier to try out the application.
demo = gr.Interface(
    fn=predict, 
    inputs=["number","number","number","number","number","number","number"], 
    outputs="textbox",
    examples=[
        [57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0],
        [15.694985541059943,175.98918151972342,0.8556228290724207,1.0,0.0,0.0,1.0]
        ],
    title="Predict Credit Card Fraud"
    )


if __name__ == "__main__":
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
