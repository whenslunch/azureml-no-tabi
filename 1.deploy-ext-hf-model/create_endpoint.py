import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
from azure.identity import ClientSecretCredential

# Load the configuration file
with open("config.json") as f:
    config = json.load(f)

name=config["workspace_name"]
subscription_id = config["subscription_id"]
resource_group = config["resource_group"]
workspace_name = config["workspace_name"]
tenant_id = config["tenant_id"]
client_id = config["client_id"]
client_secret = config["client_secret"]

credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

# get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
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

registered_model = ml_client.models.get(name="flux1-dev-nf4-2", version="1")

print("Model gotten")

env = ml_client.environments.get("basic-gpu-inference-env", version="2")

deployment = ManagedOnlineDeployment(
    name="brown",
    endpoint_name=endpoint_name,
    model=registered_model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./scripts", scoring_script="score.py"
    ),
    instance_type="Standard_NC40ads_H100_v5",
    instance_count=1,
)

# Create or update the Endpoint
print("Creating or updating the endpoint...")
endpoint_poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
endpoint_res = endpoint_poller.result()  # Waits for the operation to complete
print("Endpoint created or updated.")

# Create or update the Deployment
print("Creating or updating the deployment...")
deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment)
deployment_res = deployment_poller.result()  # Waits for the operation to complete
print("Deployment created or updated.")