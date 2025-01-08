import json
# from azureml.core import Workspace

# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, ClientSecretCredential

# Load the configuration file
with open("config.json") as f:
    config = json.load(f)

name=config["workspace_name"]
subscription_id = config["subscription_id"]
resource_group_name = config["resource_group"]
workspace_name = config["workspace_name"]
uami_id = config["uami_id"]
tenant_id = config["tenant_id"]
client_id = config["client_id"]
client_secret = config["client_secret"]

#credential = ManagedIdentityCredential(client_id=uami_id)
credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

# get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    #DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)

endpoint_name = "endpt-flux1-dev-nf4"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="endpt-flux-dev-nf4-pkg",
    auth_mode="key"
)

print("Endpoint created")

models = ml_client.models.list()
for model in models:
    print(model.name)

registered_model = ml_client.models.get(name="flux1-dev-nf4", version="1")

print("Model gotten")

env = Environment(
     #conda_file="./conda.yaml",
    #image="mcr.microsoft.com/azureml/foundation=model-inference:latest",
    image="mcr.microsoft.com/azureml/mlflow-huggingface-ubuntu20.04-py38-nvidia-gpu-inference:20240122.v1"
    )

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=registered_model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./", scoring_script="score.py"
    ),
    instance_type="Standard_NC40ads_H100_v5",
    instance_count=1,
)

ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)

ml_client.online_deployments.begin_create_or_update(blue_deployment, local=True)

ml_client.online_endpoints.get(name=endpoint_name, local=True)

ml_client.online_deployments.begin_create_or_update(
    deployment=blue_deployment, local=True
)


