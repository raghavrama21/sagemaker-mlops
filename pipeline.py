# %%
import boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.estimator import Estimator

from sagemaker.image_uris import retrieve

from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.experiments.run import Run

EXPERIMENT_NAME = "kpi1-home-credit-tuning"
ROLE = "arn:aws:iam::846754631130:role/kpi1-sagemaker-execution-role"
BUCKET = "kpi1-mlops-846754631130"
PIPELINE_NAME = "kpi1-home-credit-pipeline"
REGION = boto3.Session().region_name
pipeline_session = PipelineSession()

XGB_IMAGE = retrieve("xgboost", REGION, version="1.7-1")
LL_IMAGE = retrieve("linear-learner", REGION, version="1.7-1")


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

xgb_estimator = Estimator(
    image_uri=XGB_IMAGE,
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=f"s3://{BUCKET}/models/xgb/",
    sagemaker_session=pipeline_session,
)

xgb_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    eval_metric="auc",
)

xgb_tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges={
        "max_depth": IntegerParameter(3, 10),
        "eta": ContinuousParameter(0.1, 0.4),
        "min_child_weight": ContinuousParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1),
    },
    max_jobs=4,
    max_parallel_jobs=2,
)

with Run(
    experiment_name=EXPERIMENT_NAME,
    run_name="first-run",
    sagemaker_session=pipeline_session,
) as run:
    step_args = xgb_tuner.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

xgb_tuning_step = TuningStep(
    name="TuneXGBoost",
    step_args=step_args,
)

ll_estimator = Estimator(
    image_uri=LL_IMAGE,
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=f"s3://{BUCKET}/models/ll/",
    sagemaker_session=pipeline_session,
)

ll_estimator.set_hyperparameters(
    predictor_type="binary_classifier",
)

ll_tuner = HyperparameterTuner(
    estimator=ll_estimator,
    objective_metric_name="validation:roc_auc_score",
    objective_type="Maximize",
    hyperparameter_ranges={
        "wd": ContinuousParameter(1e-7, 1),
        "l1": ContinuousParameter(1e-7, 1),
        "learning_rate": ContinuousParameter(1e-5, 1),
        "mini_batch_size": IntegerParameter(100, 5000),
        "use_bias": CategoricalParameter([True, False]),
    },
    max_jobs=4,
    max_parallel_jobs=2,
)

with Run(
    experiment_name=EXPERIMENT_NAME,
    run_name="LL-run",  # remember: no underscores
    sagemaker_session=pipeline_session,
) as run:
    ll_step_args = ll_tuner.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

ll_tuning_step = TuningStep(name="TuneLinearLearner", step_args=ll_step_args)

pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[preprocess_step, xgb_tuning_step, ll_tuning_step],
)
if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print(f"Started execution: {execution.arn}")
    execution.wait()
    print(execution.list_steps())
