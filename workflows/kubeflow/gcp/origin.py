import kfp.dsl as dsl 
from kfp.gcp import use_gcp_secret
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
def pipeline_definition():

    tools = dsl.ContainerOp(
        name="tools",
        image=f"hydrosphere/workshop-tools:{tag}",
    ).apply(use_config_map("mnist-workflow"))

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