import os
import logging
import json
import numpy
import joblib
import torch 
from io import BytesIO
import base64
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel
from PIL import Image
from flask import Response

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
    
    # nf4_model_id = "sayakpaul/flux.1-dev-nf4-pkg"
    # text_encoder_2 = T5EncoderModel.from_pretrained(nf4_model_id, subfolder="text_encoder_2", torch_dtype=torch.float16, cache_dir=model_path)
    # transformer = FluxTransformer2DModel.from_pretrained(nf4_model_id, subfolder="transformer", torch_dtype=torch.float16, cache_dir=model_path)

    #pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", text_encoder_2=text_encoder_2, transformer=transformer, torch_dtype=torch.float16, cache_dir=model_path)
    
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    logging.info("Model initialized")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    data = json.loads(raw_data)

    logging.info("Request received")
    #logging.info("Request data: " + data)
    print("Request received")
    print(data)

    try:
        image = pipe(
            prompt = data["prompt"],
            guidance_scale = data["guidance_scale"],
            num_inference_steps = data["num_inference_steps"],
            max_sequence_length = data["max_sequence_length"],
            generator=torch.Generator("cuda").manual_seed(data["generator_seed"])
        ).images[0]

        # Convert the PIL image to PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Base64-encode the bytes for JSON transport
        #encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return Response(image_bytes, status=200, mimetype='image/png')

        # # Return as JSON
        # return json.dumps({"image": encoded_image})

    except Exception as e:
        error = str(e)
        logging.error("Error during image generation:", exc_info=True)
        return Response(f"An error occurred: {str(e)}", status=500, mimetype='text/plain')
        #return json.dumps({"error": error}) 
