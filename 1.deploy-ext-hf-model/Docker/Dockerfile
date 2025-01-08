#FROM mcr.microsoft.com/azureml/mlflow-huggingface-ubuntu20.04-py38-nvidia-gpu-inference:20240122.v1

FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:20241216.v1

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt
