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


# set up the data asset

data_path = "./data/input1.json"
data_asset_name = "t2igendatafile"

try:
    t2i_data_object = Data(
        path=data_path,
        type=AssetTypes.URI_FILE,
        description="Input data for text to image generation",
        name=data_asset_name
    )
    print(f"Data asset '{data_asset_name}' created.")
except Exception as e:
    print(f"Failed to create data asset: {e}")
    raise

try:
    ml_client.data.create_or_update(t2i_data_object)
    print(f"Data asset '{data_asset_name}' uploaded.")
except Exception as e:
    print(f"Failed to upload data asset: {e}")
    raise