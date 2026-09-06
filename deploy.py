import boto3
from sagemaker.model import ModelPackage
from sagemaker.session import Session, production_variant

ROLE = "arn:aws:iam::846754631130:role/kpi1-sagemaker-execution-role"
REGION = boto3.Session().region_name

XGB_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-east-1:846754631130:model-package/kpi1-xgb-variant/1"
)
LL_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-east-1:846754631130:model-package/kpi1-linear-variant/1"
)

ENDPOINT_NAME = "my-first-endpoint-rr"
INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1
XGB_WEIGHT = 50
LL_WEIGHT = 50

sagemaker_session = Session(boto3.Session(region_name=REGION))

xgb_model = ModelPackage(
    role=ROLE,
    model_package_arn=XGB_MODEL_PACKAGE_ARN,
    sagemaker_session=sagemaker_session,
    name="xgb-endpoint",
)
ll_model = ModelPackage(
    role=ROLE,
    model_package_arn=LL_MODEL_PACKAGE_ARN,
    sagemaker_session=sagemaker_session,
    name="ll-endpoint",
)

xgb_model.create(instance_type=INSTANCE_TYPE)
ll_model.create(instance_type=INSTANCE_TYPE)

xgb_variant = production_variant(
    model_name=xgb_model.name,
    instance_type=INSTANCE_TYPE,
    initial_instance_count=INSTANCE_COUNT,
    variant_name="XGBVariant",  # match the naming style used for steps elsewhere
    initial_weight=XGB_WEIGHT,
)
ll_variant = production_variant(
    model_name=ll_model.name,
    instance_type=INSTANCE_TYPE,
    initial_instance_count=INSTANCE_COUNT,
    variant_name="LLVariant",
    initial_weight=LL_WEIGHT,
)

if __name__ == "__main__":
    endpoint_name = sagemaker_session.endpoint_from_production_variants(
        name=ENDPOINT_NAME,
        production_variants=[
            xgb_variant,
            ll_variant,
        ],
        wait=True,
    )
    print(f"Endpoint in service: {endpoint_name}")
