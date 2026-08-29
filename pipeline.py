from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

ROLE = "arn:aws:iam::846754631130:role/kpi1-sagemaker-execution-role"
BUCKET = "kpi1-mlops-846754631130"
PIPELINE_NAME = "kpi1-home-credit-pipeline"

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
)

preprocess_step = ProcessingStep(
    name="PreprocessHomeCreditData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/raw/application_train.csv",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=f"s3://{BUCKET}/processed/train",
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/validation",
            destination=f"s3://{BUCKET}/processed/validation",
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=f"s3://{BUCKET}/processed/test",
        ),
    ],
    code="preprocess.py",
)

pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[preprocess_step],
)

# NEXT: XGBoost tuning branch (HyperparameterTuner + TuningStep + Experiments Run),
# then the matching Linear Learner branch, then ConditionStep + RegisterModel for each.
# See README.md.

if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print(f"Started execution: {execution.arn}")
    execution.wait()
    print(execution.list_steps())
