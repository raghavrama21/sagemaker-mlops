# Multi-Variant SageMaker Endpoint for Credit Default Risk

A production-style MLOps pipeline on Amazon SageMaker: two independently
tuned models trained under one SageMaker Pipeline, registered through the
Model Registry, and served behind a single endpoint as traffic-split
production variants — with auto-scaling and drift monitoring on top.

## Overview

This project builds a binary classifier for credit default risk (Home
Credit Default Risk dataset, ~307K rows, ~8% positive rate) and wraps it in
the infrastructure needed to run it like a real service, not a notebook
demo:

- A single **SageMaker Pipeline** handles preprocessing once and branches
  into two independent training/tuning paths.
- **XGBoost** and **Linear Learner** are tuned in parallel via
  `HyperparameterTuner`, with every trial tracked under one SageMaker
  Experiment for side-by-side comparison.
- Each branch is gated by a validation-AUC `ConditionStep` before
  registering to its own Model Package Group — different algorithms are
  treated as different model lineages, not versions of the same model.
- Approved models deploy as **production variants behind one endpoint**
  with a configurable traffic split, each with its own auto-scaling policy.
- **Model Monitor** watches for input drift (PSI) against a training
  baseline and fires a CloudWatch alarm through SNS.
- **Locust**, driving SigV4-signed `invoke_endpoint` calls, load-tests the
  endpoint against a p50 ≤ 50ms @ 100 RPS target, cross-checked against
  CloudWatch's `ModelLatency` metric.

## Architecture

```
                    Preprocess (shared)
              ┌────────────┴────────────┐
       Tune: XGBoost (AMT)        Tune: Linear Learner (AMT)
       tracked under one SageMaker Experiment
              │                          │
       ConditionStep (val AUC)    ConditionStep (val AUC)
              │                          │
       Register → Model Package    Register → Model Package
       Group "kpi1-xgb-variant"    Group "kpi1-linear-variant"
```

Outside the training pipeline, both approved model packages deploy as
production variants behind one endpoint, each with target-tracking
auto-scaling on `SageMakerVariantInvocationsPerInstance`.

## Why two algorithms

XGBoost and Linear Learner are genuinely different model families, not two
configurations of the same model — that makes the eventual A/B / traffic-split
comparison meaningful rather than cosmetic. Both are SageMaker built-in
algorithms, so no custom container is required for training.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle). Target column `TARGET` (binary, ~8% positive rate). Preprocessing
drops the row-ID column, imputes missing values (median for numeric,
explicit "Missing" category for categorical), one-hot encodes categoricals,
and writes a stratified 70/15/15 train/validation/test split in the
header-less, label-first CSV format SageMaker's built-in XGBoost and Linear
Learner containers require.

## Stack

- **SageMaker**: Pipelines, Processing, built-in XGBoost & Linear Learner,
  Automatic Model Tuning, Experiments, Model Registry, Model Monitor,
  endpoint auto-scaling
- **boto3** / **sagemaker SDK v2**
- **pandas**, **scikit-learn** for preprocessing
- **Locust** for load testing (SigV4-signed requests against the live
  endpoint)
- **uv** for dependency management

## Project status

- [x] AWS environment provisioned (IAM execution role, S3 bucket, SageMaker
      Studio domain)
- [x] Preprocessing implemented and validated locally and as a SageMaker
      Processing job (`preprocess.py`)
- [x] Pipeline definition (`pipeline.py`) with the preprocessing step
      running end-to-end against AWS
- [ ] XGBoost and Linear Learner tuning branches
- [ ] Conditional model registration to separate Model Package Groups
- [ ] Multi-variant endpoint deployment with traffic splitting
- [ ] Auto-scaling policies per variant
- [ ] Model Monitor drift detection with CloudWatch/SNS alerting
- [ ] Locust load test against latency target

## Running the pipeline

```bash
uv sync
uv run pipeline.py
```

This upserts and starts the SageMaker Pipeline defined in `pipeline.py`
against the AWS account/region configured for the execution role.
