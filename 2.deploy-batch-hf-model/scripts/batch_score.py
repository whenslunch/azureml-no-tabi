import os
import logging
import json
import torch 
from diffusers import FluxPipeline
import pandas as pd
from typing import List

def init():

    global output_path
    global pipe

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "saved_model"
    )

    logging.info("Model path: " + model_path)
    print("Model path: " + model_path)

    output_path = os.getenv("AZUREML_BI_OUTPUT_PATH")
    logging.info("Output path: " + output_path)
    print("Output path: " + output_path)
    
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    logging.info("Model initialized")
    print("Model initialized")

def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    Each file in the input directory contains json config data. They are used to generate images with the Flux pipeline.
    Images are saved to the output path, which is the workspaceblobstore by default. 
    The path to the images are appended to a results file and that's what gets returned.
    """

    results = []
    for idx, file_path in enumerate(mini_batch, start=1):
        try:
            logging.info(f"Processing file {idx}/{len(mini_batch)}: {file_path}")
            print(f"Processing file {idx}/{len(mini_batch)}: {file_path}")
            
            # Open and load JSON data
            with open(file_path, "r") as f:
                data = json.load(f)

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

            # Save the image as PNG file
            image.save(output_dir, format="PNG")
            
            # Append the file path to results
            results.append({"image_path": output_dir})
            logging.info(f"Successfully processed and saved image to: {output_path}")
            print(f"Successfully processed and saved image to: {output_dir}")
        
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
