from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from azure.identity import ClientSecretCredential


import json

with open("config.json") as f:
    config = json.load(f)

subscription_id = config["subscription_id"]
resource_group = config["resource_group"]
workspace_name = config["workspace_name"]
tenant_id = config["tenant_id"]
client_id = config["client_id"]
client_secret = config["client_secret"]


# create the Azure ML Client with a service principal

try:
    credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    print(f"MLClient created for {workspace_name}")
except Exception as e:
    print(f"Failed to create MLClient: {e}")
    raise

endpoint_name = "flux-nf4-batch"
deployment_name = "nf4-batch-deployment"
data_asset_name = "t2igendata"

try:
    t2i_data_asset = ml_client.data.get(name=data_asset_name, label="latest")
    print(f"Data asset '{data_asset_name}:{t2i_data_asset.id}' found.")
except Exception as e:
    print(f"Data asset not found: {e}")
    raise

input = Input(path=t2i_data_asset.id)

# Query the endpoint

try:
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name,
        input=input
    )
    print(f"Job '{job.name}' created.")
except Exception as e:
    print(f"Failed to create job: {e}")
    raise



