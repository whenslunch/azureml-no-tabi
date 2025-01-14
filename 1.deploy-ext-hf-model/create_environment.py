from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import ClientSecretCredential
import json

with open("config.json") as f:
    config = json.load(f)   

tenant_id = config["tenant_id"]
client_id = config["client_id"]
client_secret = config["client_secret"]

credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

# Initialize MLClient
ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"]
)

# Define Azure ML Environment
environment = Environment(
    name="basic-gpu-inference-env",
    image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:20241216.v1",
    conda_file="config.yml",
    description="Environment for Azure ML deployment"
)

# Register the Environment
ml_client.environments.create_or_update(environment)

# Query and list all environments in the workspace
environments = ml_client.environments.list()
print("Environments in the workspace:")
for env in environments:
    print(f"- {env.name}")