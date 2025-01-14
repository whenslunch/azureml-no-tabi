import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential, ClientSecretCredential, CertificateCredential

# Load the configuration file
with open("config.json") as f:
    config = json.load(f)

subscription_id = config["subscription_id"]
resource_group = config["resource_group"]
workspace_name = config["workspace_name"]
uami_id = config["uami_id"]
tenant_id = config["tenant_id"]
client_id = config["client_id"]
client_secret = config["client_secret"]
certificate_path = config["certificate_path"]
certificate_password = config["certificate_password"]

# MI can't be used from local machine because it can't access IMDS (Instance Metadata Service) at 169.254.169.254 to get authed.
# credential = ManagedIdentityCredential(client_id=uami_id)

# Either SP methods can be used from local machine.
# But have to enable key-auth on the storage account.
credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

registered_model = ml_client.models.create_or_update(
    Model(
        name="flux1-dev-nf4-2",
        path="./saved_model",
        description="Flux Dev NF4 saved model trial",
        type="custom_model"
    )
)

print("Model registered:", registered_model.name, "version:", registered_model.version)