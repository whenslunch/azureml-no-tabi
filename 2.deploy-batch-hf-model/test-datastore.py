from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Datastore

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

datastore = ml_client.datastores.get(name="workspaceblobstore")
print(f"Datastore: {datastore.name}")
print(f"Datastore ID: {datastore.id}")
print(f"Datastore type: {datastore.type}")
print(f"Datastore properties: {datastore.properties}")
print(f"Datastore description: {datastore.description}")
print(f"Datastore account name: {datastore.account_name}")
print(f"Datastore endpoint: {datastore.endpoint}")
print(f"Datastore base path: {datastore.base_path}")
