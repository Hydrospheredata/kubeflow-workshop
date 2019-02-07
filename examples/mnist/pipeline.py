import kfp.dsl as dsl 
import kubernetes.client.models as k8s


@dsl.pipeline(name="mnist", description="MNIST classifier")
def pipeline_definition(
    hydrosphere_name="local",
    hydrosphere_address="http://hydro-serving-sidecar-serving.kubeflow.svc.cluster.local:8080",
    data_directory='/data/mnist',
    models_directory="/models/mnist",
    learning_rate="0.01",
    learning_steps="5000",
    batch_size="256",
    warmpup_count="100",
    model_name="mnist",
    application_name="mnist-app",
    signature_name="predict",
    acceptable_accuracy="0.90",
):

    data_pvc = k8s.V1PersistentVolumeClaimVolumeSource(claim_name="data")
    models_pvc = k8s.V1PersistentVolumeClaimVolumeSource(claim_name="models")
    data_volume = k8s.V1Volume(name="data", persistent_volume_claim=data_pvc)
    models_volume = k8s.V1Volume(name="models", persistent_volume_claim=models_pvc)
    data_volume_mount = k8s.V1VolumeMount(
        mount_path="{{workflow.parameters.data-directory}}", name="data")
    models_volume_mount = k8s.V1VolumeMount(
        mount_path="{{workflow.parameters.models-directory}}", name="models")
    
    hydrosphere_address_env = k8s.V1EnvVar(
        name="CLUSTER_ADDRESS", value="{{workflow.parameters.hydrosphere-address}}")
    hydrosphere_name_env = k8s.V1EnvVar(
        name="CLUSTER_NAME", value="{{workflow.parameters.hydrosphere-name}}")
    data_directory_env = k8s.V1EnvVar(
        name="MNIST_DATA_DIR", value="{{workflow.parameters.data-directory}}")
    models_directory_env = k8s.V1EnvVar(
        name="MNIST_MODELS_DIR", value="{{workflow.parameters.models-directory}}")
    model_name_env = k8s.V1EnvVar(
        name="MODEL_NAME", value="{{workflow.parameters.model-name}}")
    application_name_env = k8s.V1EnvVar(
        name="APPLICATION_NAME", value="{{workflow.parameters.application-name}}")
    signature_name_env = k8s.V1EnvVar(
        name="SIGNATURE_NAME", value="{{workflow.parameters.signature-name}}")
    acceptable_accuracy_env = k8s.V1EnvVar(
        name="ACCEPTABLE_ACCURACY", value="{{workflow.parameters.acceptable-accuracy}}")
    learning_rate_env = k8s.V1EnvVar(
        name="LEARNING_RATE", value="{{workflow.parameters.learning-rate}}")
    learning_steps_env = k8s.V1EnvVar(
        name="LEARNING_STEPS", value="{{workflow.parameters.learning-steps}}")
    batch_size_env = k8s.V1EnvVar(
        name="BATCH_SIZE", value="{{workflow.parameters.batch-size}}")
    warmup_count_env = k8s.V1EnvVar(
        name="WARMUP_IMAGES_AMOUNT", value="{{workflow.parameters.warmpup-count}}")



    # 1. Download MNIST data
    download = dsl.ContainerOp(
        name="download",
        image="tidylobster/mnist-pipeline-download:latest")
    download.add_volume(data_volume)
    download.add_volume_mount(data_volume_mount)
    download.add_env_variable(data_directory_env)
    

    # 2. Train and save a MNIST classifier using Tensorflow
    train = dsl.ContainerOp(
        name="train",
        image="tidylobster/mnist-pipeline-train:latest")
    train.after(download)
    train.set_memory_request('2G')
    train.set_cpu_request('1')

    train.add_volume(data_volume)
    train.add_volume(models_volume) 
    train.add_volume_mount(data_volume_mount)
    train.add_volume_mount(models_volume_mount)
    train.add_env_variable(data_directory_env)
    train.add_env_variable(models_directory_env)
    train.add_env_variable(learning_rate_env)
    train.add_env_variable(learning_steps_env)
    train.add_env_variable(batch_size_env)

    # 3. Upload trained model to the cluster
    upload = dsl.ContainerOp(
        name="upload",
        image="tidylobster/mnist-pipeline-upload:latest",
        file_outputs={"model_version": "/model_version.txt"})
    upload.after(train)
    
    upload.add_volume(models_volume) 
    upload.add_volume_mount(models_volume_mount)
    upload.add_env_variable(models_directory_env)
    upload.add_env_variable(model_name_env)
    upload.add_env_variable(hydrosphere_name_env)
    upload.add_env_variable(hydrosphere_address_env)

    # 4. Deploy application
    deploy = dsl.ContainerOp(
        name="deploy",
        image="tidylobster/mnist-pipeline-deploy:latest",
        arguments=[upload.outputs["model_version"]])
    deploy.after(upload)

    deploy.add_env_variable(hydrosphere_name_env)
    deploy.add_env_variable(hydrosphere_address_env)
    deploy.add_env_variable(application_name_env)
    deploy.add_env_variable(model_name_env)

    # 5. Test the model 
    test = dsl.ContainerOp(
        name="test",
        image="tidylobster/mnist-pipeline-test:latest")
    test.after(deploy)

    test.add_volume(data_volume) 
    test.add_volume_mount(data_volume_mount)
    test.add_env_variable(data_directory_env)
    test.add_env_variable(hydrosphere_address_env)
    test.add_env_variable(application_name_env)
    test.add_env_variable(signature_name_env)
    test.add_env_variable(warmup_count_env)
    test.add_env_variable(acceptable_accuracy_env)

    # 6. Clean environment
    clean = dsl.ContainerOp(
        name="clean",
        image="tidylobster/mnist-pipeline-clean:latest")
    clean.after(test)

    clean.add_volume(data_volume)
    clean.add_volume_mount(data_volume_mount)
    clean.add_env_variable(data_directory_env)
    clean.add_volume(models_volume) 
    clean.add_volume_mount(models_volume_mount)
    clean.add_env_variable(models_directory_env)
    

if __name__ == "__main__":
    import sys
    import kfp.compiler as compiler
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py output_file"); sys.exit(-1)

    filename = sys.argv[1]
    compiler.Compiler().compile(pipeline_definition, filename)