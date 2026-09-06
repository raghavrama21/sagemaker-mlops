import random
import boto3
from collections import Counter
from datetime import datetime, timezone

ENDPOINT_NAME = "my-first-endpoint-rr"
REGION = boto3.Session().region_name
NUM_REQUESTS = 1000
NUM_FEATURES = 250
BUCKET = "kpi1-mlops-846754631130"
OUTPUT_KEY = (
    f"test-results/endpoint-test-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
)

runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def fake_row():
    values = [random.random() for _ in range(NUM_FEATURES)]
    return ",".join(str(v) for v in values)


variant_counts = Counter()
result_rows = ["request_index,variant,prediction"]

for i in range(NUM_REQUESTS):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=fake_row(),
    )
    variant = response["InvokedProductionVariant"]
    prediction = response["Body"].read().decode("utf-8").strip()
    variant_counts[variant] += 1
    result_rows.append(f'{i},{variant},"{prediction}"')

s3.put_object(
    Bucket=BUCKET,
    Key=OUTPUT_KEY,
    Body="\n".join(result_rows).encode("utf-8"),
)

print(variant_counts)
print(f"Predictions written to s3://{BUCKET}/{OUTPUT_KEY}")
