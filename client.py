import kfp
import namesgenerator

client = kfp.Client("localhost:8889")  # ml-pipeline endpoint in kubeflow namespace
experiment_name='MNIST Showreal'
pipeline_filename = 'pipeline.tar.gz'
run_name = namesgenerator.get_random_name()

try:
    experiment_id = client.get_experiment(experiment_name=experiment_name).id
except:
    experiment_id = client.create_experiment(experiment_name).id

# Submit a pipeline run
run_result = client.run_pipeline(
    experiment_id, 
    run_name, 
    pipeline_filename, 
    {
        "hydrosphere-address": "https://dev.k8s.hydrosphere.io",
        "requests_delay": "2",
    }
)