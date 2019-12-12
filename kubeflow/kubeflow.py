import kfp, sys
import namesgenerator
import argparse


# Parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--file', help='Compiled pipeline file [.tar.gz, .yaml, .zip]', required=True)
parser.add_argument(
    '-e', '--experiment', help='Experiment name to run pipeline on', default='Default')
parser.add_argument(
    '-r', '--run-name', help="Run name", default=None)
parser.add_argument(
    '-k', '--kubeflow', help="Host, where Kubeflow instance is running", required=True)
args = parser.parse_args()


# Create client
client = kfp.Client(args.kubeflow)
run_name = 'mnist_' + namesgenerator.get_random_name() if not args.run_name else args.run_name

try:
    experiment_id = client.get_experiment(experiment_name=args.experiment).id
except:
    experiment_id = client.create_experiment(args.experiment).id


# Submit a pipeline run
result = client.run_pipeline(experiment_id, run_name, args.file, params={"drift_detector_steps": "500"})
print(result)