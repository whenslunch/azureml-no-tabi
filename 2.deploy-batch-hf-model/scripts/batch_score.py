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
    global output_path
    global pipe

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "saved_model"
    )

    print("**********************************")
    print("**********************************")
    print("*******     MODEL INIT     *******")
    print("**********************************")
    print("**********************************")

    logging.info("Model path: " + model_path)
    print("Model path: " + model_path)

    output_path = os.getenv("AZUREML_BI_OUTPUT_PATH")
    print("Output path: " + output_path)
    
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
    for idx, file_path in enumerate(mini_batch, start=1):
        try:
            logging.info(f"Processing file {idx}/{len(mini_batch)}: {file_path}")
            print(f"Processing file {idx}/{len(mini_batch)}: {file_path}")
            
            print(f"Opening and loading JSON data from: {file_path}")
            # Open and load JSON data
            with open(file_path, "r") as f:
                data = json.load(f)

            print(f"Generating image using Flux pipeline")

            # Generate image using the pipeline
            image = pipe(
                prompt=data["prompt"],
                guidance_scale=data["guidance_scale"],
                num_inference_steps=data["num_inference_steps"],
                max_sequence_length=data["max_sequence_length"],
                generator=torch.Generator("cuda").manual_seed(data["generator_seed"])
            ).images[0]
            
            # Generate a unique filename to avoid overwriting
            unique_filename = f"generated-image-{idx}.png"
            output_dir = os.path.join(output_path, unique_filename)

            print(f"Saving image to: {output_dir}")
            
            # Save the image as PNG file
            image.save(output_dir, format="PNG")

            
            # Append the file path to results
            results.append({"image_path": output_dir})
            logging.info(f"Successfully processed and saved image to: {output_path}")
            print(f"Successfully processed and saved image to: {output_dir}")
        
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            print(f"File not found: {file_path}")
            continue  # Skip to the next file in the batch
        
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in file: {file_path}")
            print(f"Invalid JSON format in file: {file_path}")
            continue
        
        except KeyError as ke:
            logging.error(f"{ke} in file: {file_path}")
            print(f"{ke} in file: {file_path}")
            continue
        
        except Exception as e:
            logging.error(f"Error processing file '{file_path}': {e}")
            print(f"Error processing file '{file_path}': {e}")
            continue

    if not results:
        logging.error("No successful mini batch items were returned from run().")
        print("No successful mini batch items were returned from run().")
        raise RuntimeError("Batch run failed: All mini batch items failed.")

    logging.info("Batch processing completed successfully.")
    print("Batch processing completed successfully.")
    return pd.DataFrame(results)

    # for file_path in mini_batch:
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
                
    #     image = pipe(
    #         prompt = data["prompt"],
    #         guidance_scale = data["guidance_scale"],
    #         num_inference_steps = data["num_inference_steps"],
    #         max_sequence_length = data["max_sequence_length"],
    #         generator=torch.Generator("cuda").manual_seed(data["generator_seed"])
    #     ).images[0]

    #     # Convert the image to PNG bytes
    #     buffer = BytesIO()
    #     image.save(buffer, format="PNG")
    #     image_bytes = buffer.getvalue()   # not saving properly as PNG, cannot open

    #     # Encode the PNG bytes to Base64 string
    #     encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    #     results.append(
    #         {
    #             "image": encoded_image
    #         }
    #     )

    # return pd.DataFrame(results)

