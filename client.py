import kfp, sys
import namesgenerator

assert len(sys.argv) == 2, \
    "You have to provide compiled pipeline resource as an argument."

client = kfp.Client("http://f7fb8c1f.kubeflow.odsc.k8s.hydrosphere.io")
experiment_name='MNIST Showreal'
pipeline_filename = sys.argv[1]
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
        "hydrosphere-address": "http://f7fb8c1f.serving.odsc.k8s.hydrosphere.io",
        "requests_delay": "2",
        
        # "recurring-run": "1",
    }
)
