import boto3

REGION = boto3.Session().region_name
ENDPOINT_NAME = "my-first-endpoint-rr"
VARIANT_NAMES = ["XGBVariant", "LLVariant"]

MIN_CAPACITY = 1  # smallest number of instances this variant can shrink to
MAX_CAPACITY = 2  # largest number it can grow to
TARGET_INVOCATIONS_PER_INSTANCE = 500  # the thermostat's target — how many requests/min is "busy enough to add an instance"?

client = boto3.client("application-autoscaling", region_name=REGION)

for variant_name in VARIANT_NAMES:
    resource_id = f"endpoint/{ENDPOINT_NAME}/variant/{variant_name}"

    client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=MIN_CAPACITY,
        MaxCapacity=MAX_CAPACITY,
    )

    client.put_scaling_policy(
        PolicyName="my-first-autoscale-policy",  # a name for this policy — pick something traceable to the variant
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": TARGET_INVOCATIONS_PER_INSTANCE,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 120,
            "ScaleOutCooldown": 60,
        },
    )
    print(f"Auto-scaling registered for {variant_name}")
