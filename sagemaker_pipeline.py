import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.step_collections import RegisterModel

# Inicialización
session = sagemaker.Session()
region = session.boto_region_name
bucket = session.default_bucket()
role = sagemaker.get_execution_role()
prefix = "trabajofinalmlops"
pipeline_session = PipelineSession()

# Parámetros del pipeline
input_data_param = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://{bucket}/{prefix}/input_data/online_shoppers_intention.csv"
)

# === Paso 1: Preprocesamiento ===
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region=region, version="1.0-1"),
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[
        ProcessingInput(source=input_data_param, destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test")
    ],
    code="code/preprocess.py"
)

# === Paso 2: Entrenamiento ===
xgb_estimator = Estimator(
    entry_point="code/train.py",
    image_uri=sagemaker.image_uris.retrieve("xgboost", region=region, version="1.7-1"),
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{bucket}/{prefix}/model",
    role=role,
    sagemaker_session=pipeline_session
)

train_step = TrainingStep(
    name="TrainXGBoostModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# === Paso 3: Evaluación ===
evaluation_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1"),
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/{prefix}/evaluation"
        )
    ],
    code="code/evaluate.py"
)

# === (Opcional) Paso 4: Registro del modelo ===
register_step = RegisterModel(
    name="RegisterXGBoostModel",
    estimator=xgb_estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="OnlineShoppersModelGroup"
)

# === Pipeline ===
pipeline = Pipeline(
    name="OnlineShoppersPipeline",
    parameters=[input_data_param],
    steps=[preprocess_step, train_step, evaluation_step, register_step],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()