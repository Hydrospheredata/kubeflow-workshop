import kfp.dsl as dsl 
from kfp.aws import use_aws_secret
from kfp.gcp import use_gcp_secret
import kubernetes.client.models as k8s
import argparse, os


@dsl.pipeline(name="MNIST", description="MNIST Workflow Example")
def pipeline_definition(
    application_name="mnist_app",
    bucket_name="gs://workshop-hydrosphere",
    model_learning_rate="0.01",
    model_epochs="10",
    model_batch_size="256",
    drift_detector_learning_rate="0.01",
    drift_detector_steps="3500",
    drift_detector_batch_size="256",
    model_name="mnist",
    model_drift_detector_name="mnist_drift_detector",
    acceptable_accuracy="0.80",
):
    """ Pipeline describes structure in which steps should be executed. """

    # 1. Sample production traffic and prepare training data
    sample = dsl.ContainerOp(
        name="sample",
        image=f"hydrosphere/mnist-pipeline-subsample:{tag}",
        file_outputs={"data_path": "/data_path.txt"},
        arguments=[
            "--application-name", application_name,
            "--bucket-name", bucket_name,
        ]
    ).apply(use_gcp_secret())

    # 2. Train MNIST classifier
    train_model = dsl.ContainerOp(
        name="train_model",
        image=f"hydrosphere/mnist-pipeline-train-model:{tag}",
        file_outputs={
            "model_path": "/model_path.txt",
            "classes": "/classes.txt",
            "accuracy": "/accuracy.txt",
            "average_loss": "/average_loss.txt",
            "global_step": "/global_step.txt",
            "loss": "/loss.txt",
            "mlflow_link": "/mlflow_link.txt",
        },
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
            "--model-name", model_name, 
            "--bucket-name", bucket_name,
        ]
    ).apply(use_gcp_secret())
    train_model.set_memory_request('1G')
    train_model.set_cpu_request('1')

    # 3. Train Drift Detector on MNIST dataset
    train_drift_detector = dsl.ContainerOp(
        name="train_drift_detector",
        image=f"hydrosphere/mnist-pipeline-train-drift-detector:{tag}",
        file_outputs={
            "model_path": "/model_path.txt",
            "classes": "/classes.txt",
            "loss": "/loss.txt",
            "mlflow_link": "/mlflow_link.txt",
        },
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps,
            "--model-name", model_drift_detector_name,
            "--bucket-name", bucket_name
        ]
    ).apply(use_gcp_secret())
    train_drift_detector.set_memory_request('2G')
    train_drift_detector.set_cpu_request('1')

    # 4. Release Drift Detector to Hydrosphere.io platform 
    release_drift_detector = dsl.ContainerOp(
        name="release_drift_detector",
        image=f"hydrosphere/mnist-pipeline-release-drift-detector:{tag}", 
        file_outputs={
            "model_version": "/model_version.txt",
            "model_link": "/model_link.txt"
        },
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--model-path", train_drift_detector.outputs["model_path"],
            "--model-name", model_drift_detector_name,
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps, 
            "--loss", train_drift_detector.outputs["loss"],
            "--classes", train_drift_detector.outputs["classes"],
            "--bucket-name", bucket_name,
        ]
    ).apply(use_gcp_secret())

    # 5. Deploy Drift Detector model as endpoint application 
    deploy_drift_detector_to_prod = dsl.ContainerOp(
        name="deploy_drift_detector_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name.txt",
            "application_link": "/application_link.txt"
        },
        arguments=[
            "--model-version", release_drift_detector.outputs["model_version"],
            "--application-name-postfix", "_app", 
            "--model-name", model_drift_detector_name,
            "--bucket-name", bucket_name,
        ],
    ).apply(use_gcp_secret())

    # 6. Release MNIST classifier with assigned metrics to Hydrosphere.io platform
    release_model = dsl.ContainerOp(
        name="release_model",
        image=f"hydrosphere/mnist-pipeline-release-model:{tag}", 
        file_outputs={
            "model_version": "/model_version.txt",
            "model_link": "/model_link.txt"
        },
        arguments=[
            "--drift-detector-app", deploy_drift_detector_to_prod.outputs["application_name"],
            "--model-name", model_name,
            "--classes", train_model.outputs["classes"],
            "--bucket-name", bucket_name, 
            "--data-path", sample.outputs["data_path"],
            "--model-path", train_model.outputs["model_path"],
            "--accuracy", train_model.outputs["accuracy"],
            "--average-loss", train_model.outputs["average_loss"],
            "--loss", train_model.outputs["loss"],
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
            "--global-step", train_model.outputs["global_step"],
            "--mlflow-link", train_model.outputs["mlflow_link"],
        ]
    ).apply(use_gcp_secret())

    # 7. Deploy MNIST classifier model as endpoint application on stage for testing purposes
    deploy_model_to_stage = dsl.ContainerOp(
        name="deploy_model_to_stage",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name.txt",
            "application_link": "/application_link.txt"
        },
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--application-name-postfix", "_stage_app", 
            "--model-name", model_name,
            "--bucket-name", bucket_name,
        ],
    ).apply(use_gcp_secret())

    # 8. Perform integration testing on the deployed staged application
    test_model = dsl.ContainerOp(
        name="test_model",
        image=f"hydrosphere/mnist-pipeline-test:{tag}", 
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--application-name", deploy_model_to_stage.outputs["application_name"], 
            "--acceptable-accuracy", acceptable_accuracy,
            "--bucket-name", bucket_name,
        ],
    ).apply(use_gcp_secret())
    test_model.set_retry(3)

    # 9. Deploy MNIST classifier model as endpoint application to production
    deploy_model_to_prod = dsl.ContainerOp(
        name="deploy_model_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name.txt",
            "application_link": "/application_link.txt"
        },
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--application-name-postfix", "_app", 
            "--model-name", model_name,
            "--bucket-name", bucket_name,
            "--mlflow-model-link", train_model.outputs["mlflow_link"],
            "--mlflow-drift-detector-link", train_drift_detector.outputs["mlflow_link"],
            "--data-path", sample.outputs["data_path"],
            "--model-path", train_model.outputs["model_path"],
            "--model-drift-detector-path", train_drift_detector.outputs["model_path"],
        ],
    ).apply(use_gcp_secret())
    deploy_model_to_prod.after(test_model)


if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys, argparse 

    # Get parameters	
    parser = argparse.ArgumentParser()	
    parser.add_argument('--tag', 
        help="Which tag of image to use, when compiling pipeline", default="latest")
    args = parser.parse_args()

    # Compile pipeline
    tag = args.tag
    compiler.Compiler().compile(pipeline_definition, "pipeline.tar.gz")
