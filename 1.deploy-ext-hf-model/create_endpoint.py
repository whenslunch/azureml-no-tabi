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

# create the Azure ML Client with a service principal
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)


# create an online endpoint
endpoint_name = "endpt-flux1-dev-nf4"

endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="endpt-flux-dev-nf4-pkg",
    auth_mode="key"
)

# select the model previously registered
registered_model = ml_client.models.get(name="flux1-dev-nf4-2", version="1")


# select the environment previously defined
env = ml_client.environments.get("basic-gpu-inference-env", version="2")

# create a deployment
deployment = ManagedOnlineDeployment(
    name="purple",
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
ml_client.online_endpoints.begin_create_or_update(endpoint)
print("Endpoint created or updated.")

# Create or update the Deployment
print("Creating or updating the deployment...")
ml_client.online_deployments.begin_create_or_update(deployment=deployment)
print("Deployment created or updated.")