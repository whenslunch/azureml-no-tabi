import os
import logging
import json
import numpy
import joblib
import torch 
from io import BytesIO
import base64
from diffusers import FluxPipeline
from PIL import Image
import pandas as pd
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global pipe
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "saved_model"
    )

    logging.info("Model path: " + model_path)
    print("Model path: " + model_path)
    
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    logging.info("Model initialized")
    print("Model initialized")

def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and run it through Flux pipeline to generate images. 
    Images then get stuffed into a DataFrame and returned.
    """
    print(f"Executing run method over batch of {len(mini_batch)} files.")
    logging.info(f"Executing run method over batch of {len(mini_batch)} files.")
    print("contents of mini_batch:")
    print(mini_batch)
    
    results = []
    for file_path in mini_batch:
        with open(file_path, "r") as f:
            data = json.load(f)
                
        image = pipe(
            prompt = data["prompt"],
            guidance_scale = data["guidance_scale"],
            num_inference_steps = data["num_inference_steps"],
            max_sequence_length = data["max_sequence_length"],
            generator=torch.Generator("cuda").manual_seed(data["generator_seed"])
        ).images[0]

        # Convert the image to PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()   # not saving properly as PNG, cannot open

        results.append(
            {
                "image": image_bytes
            }
        )

    return pd.DataFrame(results)

