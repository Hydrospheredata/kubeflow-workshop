import kfp

client = kfp.Client("localhost:8889")
experiment_name='MNIST Showreal'

try:
    experiment_id = client.get_experiment(experiment_name=experiment_name).id
except:
    experiment_id = client.create_experiment(experiment_name).id

pipeline_filename = 'pipeline.tar.gz'

# Submit a pipeline run
run_name = '04'
run_result = client.run_pipeline(
    experiment_id, 
    run_name, 
    pipeline_filename, 
    {
        "hydrosphere-address": "https://dev.k8s.hydrosphere.io",
        "learning-steps": "1000",
        "requests_delay": "2",

    }
)