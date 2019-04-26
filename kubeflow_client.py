import kfp, sys
import namesgenerator
import argparse


# Parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--file', help='Compiled pipeline file [.tar.gz, .yaml, .zip]', required=True)
parser.add_argument(
    '-e', '--experiment', help='Experiment name to run pipeline on', default='MNIST Showreal')
parser.add_argument(
    '-r', '--run-name', help="Run name", default=None)
parser.add_argument(
    '-n', '--namespace', help="Namespace, where kubeflow and serving are running", required=True)
args = parser.parse_args()
arguments = args.__dict__


# Create client
client = kfp.Client(f"http://{arguments['namespace']}.kubeflow.odsc.k8s.hydrosphere.io")
run_name = namesgenerator.get_random_name() if not arguments["run_name"] else arguments["run_name"]

try:
    experiment_id = client.get_experiment(experiment_name=arguments["experiment"]).id
except:
    experiment_id = client.create_experiment(arguments["experiment"]).id


# Submit a pipeline run
result = client.run_pipeline(
    experiment_id, run_name, arguments["file"],
    {
        "learning-rate": "0.01",
        "batch-size": "256",
        "epochs": "10",
        "hydrosphere-address": f"http://{arguments['namespace']}.serving.odsc.k8s.hydrosphere.io",
        "acceptable-accuracy": "0.90",
    }
)
