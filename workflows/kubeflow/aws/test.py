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
        "sagemaker.role",
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

    dsl.ContainerOp(
        name="tune",
        image=f"943173312784.dkr.ecr.eu-central-1.amazonaws.com/mnist-pipeline-tune-model:{tag}",
    ) \
        .apply(use_aws_secret()) \
        .apply(use_config_map("mnist-workflow")) \
        .add_env_variable(k8s.V1EnvVar(name='AWS_DEFAULT_REGION', value='eu-central-1'))


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