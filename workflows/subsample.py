import kfp.dsl as dsl 
from kfp.aws import use_aws_secret
from kubernetes import client as k8s
import argparse, os


def use_config_map(name, mount_path="/etc/config"):
    """ 
    Mounts ConfigMap defined on the cluster to the running step under specified 
    path as a file with corresponding value. 
    
    Parameters
    ----------
    name: str
        Name of the ConfigMap, defined in current namespace.
    mount_path: str
        Path, where to mount ConfigMap to. 

    Returns
    -------
    func
        A function, which have to be applied to the ContainerOp. 
    """
    
    key_path_mapper = [
        "postgres.host",
        "postgres.port",
        "postgres.user",
        "postgres.pass",
        "postgres.dbname",
        "uri.mnist",
        "uri.mlflow",
        "uri.hydrosphere",
        "uri.reqstore",
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


def apply_config_map_and_aws_secret(op):
    return op.apply(use_config_map("mnist-workflow")).apply(use_aws_secret())


@dsl.pipeline(name="MNIST", description="MNIST Workflow Example")
def pipeline_definition(
    target_application_name="kubeflow-mnist-app",
    sample_limit="10000",
    train_part="0.7",
    validation_part="0.1",
    model_learning_rate="0.005",
    model_epochs="50",
    model_batch_size="128",
    drift_detector_learning_rate="0.005",
    drift_detector_steps="5400",
    drift_detector_batch_size="128",
    model_drift_detector_name="kubeflow-mnist-drift-detector",
    model_name="kubeflow-mnist",
    acceptable_accuracy="0.80",
    test_sample_size="200",
):
    """ 
    Pipeline describes structure in which steps should be executed. 
    
    Parameters
    ----------
    target_application_name: str

    sample_limit: str

    train_part: str

    validation_part: str

    model_learning_rate: str
        Learning rate, used for training a classifier.
    model_epochs: str
        Amount of epochs, during which a classifier will be trained.
    model_batch_size: str
        Batch size, used for training a classifier.
    drift_detector_learning_rate: str
        Learning rate, used for training an autoencoder.
    drift_detector_steps: str
        Amount of steps, during which an autoencoder will be trained.
    drift_detector_batch_size: str
        Batch size, used for training an autoencoder.
    model_name: str
        Name of the classifier, which will be used for deployment.
    model_drift_detector_name: str
        Name of the autoencoder, which will be used for deployment.
    acceptable_accuracy: str
        Accuracy level indicating the final acceptable performance of the model 
        in the evaluation step, which will let let model to be either deployed 
        to production or cause workflow execution to fail. 
    """

    # Configure all steps to have ConfigMap and use aws secret
    dsl.get_pipeline_conf().add_op_transformer(apply_config_map_and_aws_secret)

    subsample = dsl.ContainerOp(
        name="subsample",
        image=f"{registry}/mnist-pipeline-subsample:{tag}",
        file_outputs={
            "output_data_path": "/output_data_path",
            "logs_path": "/logs_path",
        },
        arguments=[
            "--output-data-path", "s3://workshop-hydrosphere/mnist/data",
            "--application-name", target_application_name,
            "--limit", sample_limit,
            "--train-part", train_part, 
            "--validation-part", validation_part
        ],
    )

    train_drift_detector = dsl.ContainerOp(
        name="train_drift_detector",
        image=f"{registry}/mnist-pipeline-train-drift-detector:{tag}",
        file_outputs={
            "logs_path": "/logs_path",
            "model_path": "/model_path",
            "loss": "/loss",
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-path", "s3://workshop-hydrosphere/mnist/model",
            "--model-name", model_drift_detector_name,
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps,
        ],
    ).set_memory_request('2G').set_cpu_request('1')

    train_model = dsl.ContainerOp(
        name="train_model",
        image=f"{registry}/mnist-pipeline-train-model:{tag}",
        file_outputs={
            "logs_path": "/logs_path",
            "model_path": "/model_path",
            "accuracy": "/accuracy",
            "num_classes": "/num_classes",
            "average_loss": "/average_loss",
            "global_step": "/global_step",
            "loss": "/loss",
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-path", "s3://workshop-hydrosphere/mnist/model",
            "--model-name", model_name,
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
        ],
    ).set_memory_request('1G').set_cpu_request('1')

    release_drift_detector = dsl.ContainerOp(
        name="release_drift_detector",
        image=f"{registry}/mnist-pipeline-release-drift-detector:{tag}", 
        file_outputs={
            "model_version": "/model_version",
            "model_uri": "/model_uri"
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-path", train_drift_detector.outputs["model_path"],
            "--model-name", model_drift_detector_name,
            "--learning-rate", drift_detector_learning_rate,
            "--batch-size", drift_detector_batch_size,
            "--steps", drift_detector_steps, 
            "--loss", train_drift_detector.outputs["loss"],
        ]
    )

    deploy_drift_detector_to_prod = dsl.ContainerOp(
        name="deploy_drift_detector_to_prod",
        image=f"{registry}/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-version", release_drift_detector.outputs["model_version"],
            '--application-name-postfix=-app', 
            "--model-name", model_drift_detector_name,
        ],
    )

    release_model = dsl.ContainerOp(
        name="release_model",
        image=f"{registry}/mnist-pipeline-release-model:{tag}", 
        file_outputs={
            "model_version": "/model_version",
            "model_uri": "/model_uri"
        },
        arguments=[
            "--drift-detector-app", deploy_drift_detector_to_prod.outputs["application_name"],
            "--model-name", model_name,
            "--data-path", subsample.outputs["output_data_path"],
            "--model-path", train_model.outputs["model_path"],
            "--accuracy", train_model.outputs["accuracy"],
            "--average-loss", train_model.outputs["average_loss"],
            "--loss", train_model.outputs["loss"],
            "--learning-rate", model_learning_rate,
            "--batch-size", model_batch_size,
            "--epochs", model_epochs,
            "--global-step", train_model.outputs["global_step"],
        ]
    )

    deploy_model_to_stage = dsl.ContainerOp(
        name="deploy_model_to_stage",
        image=f"{registry}/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-version", release_model.outputs["model_version"],
            '--application-name-postfix=-stage-app',  
            "--model-name", model_name,
        ],
    )

    test_model = dsl.ContainerOp(
        name="test_model",
        image=f"{registry}/mnist-pipeline-test:{tag}", 
        file_outputs={
            "integration_test_accuracy": "/integration_test_accuracy",
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--application-name", deploy_model_to_stage.outputs["application_name"], 
            "--acceptable-accuracy", acceptable_accuracy,
            "--sample-size", test_sample_size,
        ],
    ).set_retry(3)

    deploy_model_to_prod = dsl.ContainerOp(
        name="deploy_model_to_prod",
        image=f"{registry}/mnist-pipeline-deploy:{tag}",  
        file_outputs={
            "application_name": "/application_name",
            "application_uri": "/application_uri"
        },
        arguments=[
            "--data-path", subsample.outputs["output_data_path"],
            "--model-version", release_model.outputs["model_version"],
            '--application-name-postfix=-app', 
            "--model-name", model_name,
        ],
    ).after(test_model)

    dsl.ContainerOp(
        name="write_output",
        image=f"{registry}/mnist-pipeline-output:{tag}", 
        arguments=[
            '--data-path', subsample.outputs["output_data_path"],
            '--model-artifacts-path', train_model.outputs["model_path"], 
            '--drift-detector-artifacts-path', train_drift_detector.outputs["model_path"],
            '--model-uri', release_model.outputs["model_uri"],
            '--drift-detector-uri', release_drift_detector.outputs["model_uri"],
            '--model-application-uri', deploy_drift_detector_to_prod.outputs["application_uri"],
            '--drift-detector-application-uri', deploy_model_to_prod.outputs["application_uri"], 
            '--integration-test-accuracy', test_model.outputs["integration_test_accuracy"],
        ]
    )


if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys, argparse 

    # Get parameters	
    parser = argparse.ArgumentParser()	
    parser.add_argument('--tag', 
        help="Which tag of image to use, when compiling pipeline", default="latest")
    parser.add_argument('--registry', 
        help="Which docker registry to use, when compiling pipeline", default="hydrosphere")
    args = parser.parse_args()
    
    tag = args.tag
    registry = args.registry
    # Compile pipeline
    compiler.Compiler().compile(pipeline_definition, "subsample.tar.gz")