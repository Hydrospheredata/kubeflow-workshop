import kfp.dsl as dsl 
from kfp.aws import use_aws_secret
import kubernetes.client.models as k8s
import argparse, os

tag = os.environ.get("TAG", "latest")


@dsl.pipeline(name="mnist", description="MNIST classifier")
def pipeline_definition(
    hydrosphere_address,
    application_name="mnist_app",
    model_name="mnist",
    model_learning_rate="0.0005",
    model_epochs="100",
    model_batch_size="256",
    autoencoder_learning_rate="0.01",
    autoencoder_steps="5000",
    autoencoder_batch_size="128",
    autoencoder_name="mnist_autoencoder",
    acceptable_accuracy="0.80",
):

    # 1. Sample production traffic
    sample = dsl.ContainerOp(
        name="sample",
        image=f"hydrosphere/mnist-pipeline-sample:{tag}",  # <-- Replace with correct docker image
        file_outputs={"data_path": "/data_path.txt"},
        arguments=[
            "--hydrosphere-address", hydrosphere_address,
            "--application-name", application_name,
        ]
    ).apply(use_aws_secret())

    # 2. Train and save a MNIST classifier using Tensorflow
    train_model = dsl.ContainerOp(
        name="train_model",
        image=f"hydrosphere/mnist-pipeline-train-model:{tag}",  # <-- Replace with correct docker image
        file_outputs={
            "accuracy": "/accuracy.txt",
            "model_path": "/model_path.txt",
            "classes": "/classes.txt",
        },
        arguments=[
            "--data-path", sample.outputs["data_path"], 
            "--learning-rate", model_learning_rate,
            "--epochs", model_epochs,
            "--batch-size", model_batch_size,
            "--hydrosphere-address", hydrosphere_address
        ]
    ).apply(use_aws_secret())
    train_model.set_memory_request('1G')
    train_model.set_cpu_request('1')

    # 3. Train and save a MNIST Autoencoder using Tensorflow
    train_autoencoder = dsl.ContainerOp(
        name="train_autoencoder",
        image=f"hydrosphere/mnist-pipeline-train-autoencoder:{tag}",  # <-- Replace with correct docker image
        file_outputs={
            "model_path": "/model_path.txt",
            "loss": "/loss.txt",
            "classes": "/classes.txt",
        },
        arguments=[
            "--data-path", sample.outputs["data_path"], 
            "--steps", autoencoder_steps, 
            "--learning-rate", autoencoder_learning_rate,
            "--batch-size", autoencoder_batch_size,
            "--hydrosphere-address", hydrosphere_address
        ]
    ).apply(use_aws_secret())
    train_autoencoder.set_memory_request('1G')
    train_autoencoder.set_cpu_request('1')

    # 4. Release trained autoencoder to the cluster
    release_autoencoder = dsl.ContainerOp(
        name="release_autoencoder",
        image=f"hydrosphere/mnist-pipeline-release-autoencoder:{tag}",  # <-- Replace with correct docker image
        file_outputs={"model_version": "/model_version.txt"},
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--model-name", autoencoder_name,
            "--models-path", train_autoencoder.outputs["model_path"],
            "--classes", train_autoencoder.outputs["classes"],
            "--loss", train_autoencoder.outputs["loss"],
            "--hydrosphere-address", hydrosphere_address,
            "--learning-rate", autoencoder_learning_rate,
            "--batch-size", autoencoder_batch_size,
            "--steps", autoencoder_steps, 
        ]
    ).apply(use_aws_secret())
    
    # 5. Deploy model to stage application
    deploy_autoencoder_to_prod = dsl.ContainerOp(
        name="deploy_autoencoder_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  # <-- Replace with correct docker image
        file_outputs={"application_name": "/application_name.txt"},
        arguments=[
            "--model-version", release_autoencoder.outputs["model_version"],
            "--application-name-postfix", "_app", 
            "--hydrosphere-address", hydrosphere_address,
            "--model-name", autoencoder_name,
        ],
    ).apply(use_aws_secret())
    
    # 6. Release trained model to the cluster
    release_model = dsl.ContainerOp(
        name="release_model",
        image=f"hydrosphere/mnist-pipeline-release-model:{tag}",  # <-- Replace with correct docker image
        file_outputs={"model_version": "/model_version.txt"},
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--model-name", model_name,
            "--models-path", train_model.outputs["model_path"],
            "--autoencoder-app", deploy_autoencoder_to_prod.outputs["application_name"],
            "--classes", train_model.outputs["classes"],
            "--accuracy", train_model.outputs["accuracy"],
            "--hydrosphere-address", hydrosphere_address,
            "--learning-rate", model_learning_rate,
            "--epochs", model_epochs,
            "--batch-size", model_batch_size,
        ]
    ).apply(use_aws_secret())

    # 7. Deploy model to stage application
    deploy_model_to_stage = dsl.ContainerOp(
        name="deploy_model_to_stage",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  # <-- Replace with correct docker image
        file_outputs={"application_name": "/application_name.txt"},
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--application-name-postfix", "_stage", 
            "--hydrosphere-address", hydrosphere_address,
            "--model-name", model_name,
        ],
    ).apply(use_aws_secret())

    # 8. Test the model via stage application
    test = dsl.ContainerOp(
        name="test",
        image=f"hydrosphere/mnist-pipeline-test:{tag}",  # <-- Replace with correct docker image
        arguments=[
            "--data-path", sample.outputs["data_path"],
            "--hydrosphere-address", hydrosphere_address,
            "--application-name", deploy_model_to_stage.outputs["application_name"],
            "--acceptable-accuracy", acceptable_accuracy,
        ],
    ).apply(use_aws_secret())
    test.set_retry(3)

    # 9. Deploy model to production application
    deploy_model_to_prod = dsl.ContainerOp(
        name="deploy_to_prod",
        image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  # <-- Replace with correct docker image
        file_outputs={"application_name": "/application_name.txt"},
        arguments=[
            "--model-version", release_model.outputs["model_version"],
            "--model-name", model_name,
            "--application-name-postfix", "_app", 
            "--hydrosphere-address", hydrosphere_address
        ],
    ).apply(use_aws_secret())
    deploy_model_to_prod.after(test)


if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys

    # Acquire parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--namespace', help="Namespace, where kubeflow and serving are running", required=True)
    args = parser.parse_args()

    # Compile pipeline
    compiler.Compiler().compile(pipeline_definition, "pipeline.tar.gz")

    # Replace hardcoded namespaces
    untar = f"tar -xvf pipeline.tar.gz"
    replace_minio = f"sed -i '' s/minio-service.kubeflow/minio-service.{args.namespace}/g pipeline.yaml"
    replace_pipeline_runner = f"sed -i '' s/pipeline-runner/{args.namespace}-pipeline-runner/g pipeline.yaml"

    process = subprocess.run(untar.split())
    process = subprocess.run(replace_minio.split())
    process = subprocess.run(replace_pipeline_runner.split())