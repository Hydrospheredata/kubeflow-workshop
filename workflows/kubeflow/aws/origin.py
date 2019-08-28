import kfp.dsl as dsl 
from kfp.aws import use_aws_secret
from kubernetes import client as k8s
import argparse, os


def use_config_map(name, mount_path="/etc/config"):
    """ Mounts ConfigMap defined on the cluster to the running step
        under specified path as a file with corresponding value. """
    
    key_path_mapper = [
        "postgres.host",
        "postgres.port",
        "postgres.user",
        "postgres.pass",
        "postgres.dbname",
        "uri.mnist",
        "uri.mlflow",
        "uri.hydrosphere",
        "default.tensorflow_runtime",
    ]

    def _use_config_map(task):
        config_map = k8s.V1ConfigMapVolumeSource(
            name=name,
            items=[k8s.V1KeyToPath(key=key, path=key) \
                for key in key_path_mapper]
        ) 
        return task \
            .add_volume(k8s.V1Volume(config_map=config_map, name=name)) \
            .add_volume_mount(k8s.V1VolumeMount(mount_path=mount_path, name=name))

    return _use_config_map


@dsl.pipeline(name="MNIST", description="MNIST Workflow Example")
def pipeline_definition(
    bucket_name="s3://workshop-hydrosphere",
    model_learning_rate="0.01",
    model_epochs="10",
    model_batch_size="256",
    drift_detector_learning_rate="0.01",
    drift_detector_steps="3500",
    drift_detector_batch_size="256",
    model_drift_detector_name="mnist_drift_detector",
    model_name="mnist",
    acceptable_accuracy="0.90",
):
    """ Pipeline describes structure in which steps should be executed. """

    # 1. Download MNIST dataset
    download = dsl.ContainerOp(
        name="download",
        image=f"hydrosphere/mnist-pipeline-download:{tag}",
        file_outputs={"data_path": "/data_path"},
        arguments=["--bucket-name", bucket_name]
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow"))

    # 2. Train MNIST classifier
    train_model = dsl.ContainerOp(
        name="train_model",
        image=f"hydrosphere/mnist-pipeline-train-model:{tag}",
        file_outputs={
            "model_path": "/model_path",
            "classes": "/classes",
            "accuracy": "/accuracy",
            "average_loss": "/average_loss",
            "global_step": "/global_step",
            "loss": "/loss",
            "mlflow_run_uri": "/mlflow_run_uri",
        },
        arguments=[
            "--data-path", download.outputs["data_path"],
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
            "--model-name", model_name, 
            "--bucket-name", bucket_name,
        ]
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow")) \
        .set_memory_request('1G') \
        .set_cpu_request('1')

    # 3. Train Drift Detector on MNIST dataset
    train_drift_detector = dsl.ContainerOp(
        name="train_drift_detector",
        image=f"hydrosphere/mnist-pipeline-train-drift-detector:{tag}",
        file_outputs={
            "model_path": "/model_path",
            "classes": "/classes",
            "loss": "/loss",
            "mlflow_run_uri": "/mlflow_run_uri",
        },
        arguments=[
            "--data-path", download.outputs["data_path"],
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps,
            "--model-name", model_drift_detector_name,
            "--bucket-name", bucket_name,
        ]
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow")) \
        .set_memory_request('2G') \
        .set_cpu_request('1')

    # 4. Release Drift Detector to Hydrosphere.io platform 
    release_drift_detector = dsl.ContainerOp(
        name="release_drift_detector",
        image=f"hydrosphere/mnist-pipeline-release-drift-detector:{tag}", 
        file_outputs={
            "model_version": "/model_version",
            "model_uri": "/model_uri"
        },
        arguments=[
            "--data-path", download.outputs["data_path"],
            "--model-path", train_drift_detector.outputs["model_path"],
            "--model-name", model_drift_detector_name,
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps, 
            "--loss", train_drift_detector.outputs["loss"],
            "--classes", train_drift_detector.outputs["classes"],
            "--bucket-name", bucket_name,
        ]
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow"))

    # 5. Deploy Drift Detector model as endpoint application 
    deploy_drift_detector_to_prod = dsl.ContainerOp(
        name="deploy_drift_detector_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--model-version", release_drift_detector.outputs["model_version"],
            "--application-name-postfix", "_app", 
            "--model-name", model_drift_detector_name,
            "--bucket-name", bucket_name,
        ],
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow"))

    # 6. Release MNIST classifier with assigned metrics to Hydrosphere.io platform
    release_model = dsl.ContainerOp(
        name="release_model",
        image=f"hydrosphere/mnist-pipeline-release-model:{tag}", 
        file_outputs={
            "model_version": "/model_version",
            "model_uri": "/model_uri"
        },
        arguments=[
            "--drift-detector-app", deploy_drift_detector_to_prod.outputs["application_name"],
            "--model-name", model_name,
            "--classes", train_model.outputs["classes"],
            "--bucket-name", bucket_name, 
            "--data-path", download.outputs["data_path"],
            "--model-path", train_model.outputs["model_path"],
            "--accuracy", train_model.outputs["accuracy"],
            "--average-loss", train_model.outputs["average_loss"],
            "--loss", train_model.outputs["loss"],
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
            "--global-step", train_model.outputs["global_step"],
            "--mlflow-run-uri", train_model.outputs["mlflow_run_uri"],
        ]
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow"))

    # 7. Deploy MNIST classifier model as endpoint application on stage for testing purposes
    deploy_model_to_stage = dsl.ContainerOp(
        name="deploy_model_to_stage",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--application-name-postfix", "_stage_app", 
            "--bucket-name", bucket_name,
            "--model-name", model_name,
        ],
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow"))

    # 8. Perform integration testing on the deployed staged application
    test_model = dsl.ContainerOp(
        name="test_model",
        image=f"hydrosphere/mnist-pipeline-test:{tag}", 
        arguments=[
            "--data-path", download.outputs["data_path"],
            "--application-name", deploy_model_to_stage.outputs["application_name"], 
            "--acceptable-accuracy", acceptable_accuracy,
            "--bucket-name", bucket_name,
        ],
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow")) \
        .set_retry(3)

    # # 9. Deploy MNIST classifier model as endpoint application to production
    deploy_model_to_prod = dsl.ContainerOp(
        name="deploy_model_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--application-name-postfix", "_app", 
            "--bucket-name", bucket_name,
            "--model-name", model_name,
            "--mlflow-model-run-uri", train_model.outputs["mlflow_run_uri"],
            "--mlflow-drift-detector-run-uri", train_drift_detector.outputs["mlflow_run_uri"],
            "--data-path", download.outputs["data_path"],
            "--model-path", train_model.outputs["model_path"],
            "--model-drift-detector-path", train_drift_detector.outputs["model_path"],
        ],
    ).apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow")) \
        .after(test_model)


if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys, argparse 

    # Get parameters	
    parser = argparse.ArgumentParser()	
    parser.add_argument('--tag', 
        help="Which tag of image to use, when compiling pipeline", default="latest")
    args = parser.parse_args()
    
    tag = args.tag
    # Compile pipeline
    compiler.Compiler().compile(pipeline_definition, "pipeline.tar.gz")