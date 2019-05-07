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


# Create client
client = kfp.Client(f"http://{args.namespace}.kubeflow.odsc.k8s.hydrosphere.io")
run_name = namesgenerator.get_random_name() if not args.run_name else args.run_name

try:
    experiment_id = client.get_experiment(experiment_name=args.experiment).id
except:
    experiment_id = client.create_experiment(args.experiment).id


# Submit a pipeline run
result = client.run_pipeline(
    experiment_id, run_name, args.file,
    {
        "hydrosphere-address": f"http://{args.namespace}.serving.odsc.k8s.hydrosphere.io",
    }
)
