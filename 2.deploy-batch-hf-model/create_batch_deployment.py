# Reference
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-model-deployments?view=azureml-api-2&tabs=python#create-a-batch-endpoint

from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.entities import BatchEndpoint, ModelBatchDeployment, ModelBatchDeploymentSettings, PipelineComponentBatchDeployment, Model, AmlCompute, Data, BatchRetrySettings, CodeConfiguration, Environment, Data
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.dsl import pipeline
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

credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

# create the Azure ML Client with a service principal
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# create compute cluster
compute_name = "gpu-batch-cluster"
if not any(filter(lambda m: m.name == compute_name, ml_client.compute.list())):
    compute_cluster = AmlCompute(
        name=compute_name,
        description="GPU cluster compute",
        size="Standard_NC40ads_H100_v5",
        min_instances=0,
        max_instances=1,
    )
    ml_client.compute.begin_create_or_update(compute_cluster).result()
    print(f"Compute cluster '{compute_name}' created successfully.")
else:
    print(f"Compute cluster '{compute_name}' already exists.")

# create endpoint
endpoint_name = "flux-nf4-batch"
endpoint = BatchEndpoint(
    name=endpoint_name,
    description="A batch endpoint for FLUX-1.Dev NF4 image generation.",
    tags={"type": "text-to-image"},
)
ml_client.begin_create_or_update(endpoint).result()

# Model and Environment now need to be created.
# For now, I will re-use the existing model and environments from the previous Task 1, Realtime Endpoint deployment.
# I will update this example to be standalone in the future.

# select the model previously registered
try:
    registered_model = ml_client.models.get(name="flux1-dev-nf4-2", version="1")
    print(f"Model '{registered_model.name}' found.")
except Exception as e:
    print(f"Model not found: {e}")
    raise

# select the environment previously defined
try:
    env = ml_client.environments.get("basic-gpu-inference-env", version="4")
    print(f"Environment '{env.name}' found.")
except Exception as e:
    print(f"Environment not found: {e}")
    raise   

# deploy the model in the environment

# configure the deployment

deployment = ModelBatchDeployment(
    name="nf4-batch-deployment",
    description="A deployment of FLUX-1.Dev NF4.",
    endpoint_name=endpoint_name,
    model=registered_model,
    code_configuration=CodeConfiguration(
        code="./scripts", scoring_script="batch_score.py"
    ),
    environment=env,
    compute=compute_name,
    settings=ModelBatchDeploymentSettings(
        max_concurrency_per_instance=1,
        mini_batch_size=10,
        instance_count=1,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
        logging_level="info",
    ),
)

# create the deployment
try:
    ml_client.begin_create_or_update(deployment).result()
    print(f"Deployment '{deployment.name}' created.")
except Exception as e:
    print(f"Deployment not created: {e}")
    raise


# deploy to the endpoint
try:
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    endpoint.defaults.deployment_name = deployment.name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint.name}' updated.")
except Exception as e:
    print(f"Endpoint not updated: {e}")
    raise

# Query the deployment
try:
    ml_client.batch_deployments.get(name=deployment.name, endpoint_name=endpoint.name)
    print(f"Deployment '{deployment.name}' found.")
except Exception as e:
    print(f"Deployment not found: {e}")
    raise

