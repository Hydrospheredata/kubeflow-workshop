import kfp.dsl as dsl 
import kubernetes.client.models as k8s


@dsl.pipeline(name="mnist", description="MNIST classifier")
def pipeline_definition(
    hydrosphere_address="{hydrosphere-instance-address}",  # <-- Replace with correct instance address
    mount_path='/storage',
    learning_rate="0.01",
    epochs="10",
    batch_size="256",
    test_amount="100",
    model_name="mnist",
    application_name="mnist-app",
    acceptable_accuracy="0.90",
    requests_delay="4",
):
    
    storage_pvc = k8s.V1PersistentVolumeClaimVolumeSource(claim_name="storage")
    storage_volume = k8s.V1Volume(name="storage", persistent_volume_claim=storage_pvc)
    storage_volume_mount = k8s.V1VolumeMount(
        mount_path="{{workflow.parameters.mount-path}}", name="storage")
    
    hydrosphere_address_env = k8s.V1EnvVar(
        name="CLUSTER_ADDRESS", value="{{workflow.parameters.hydrosphere-address}}")
    mount_path_env = k8s.V1EnvVar(
        name="MOUNT_PATH", value="{{workflow.parameters.mount-path}}")
    model_name_env = k8s.V1EnvVar(
        name="MODEL_NAME", value="{{workflow.parameters.model-name}}")
    application_name_env = k8s.V1EnvVar(
        name="APPLICATION_NAME", value="{{workflow.parameters.application-name}}")
    acceptable_accuracy_env = k8s.V1EnvVar(
        name="ACCEPTABLE_ACCURACY", value="{{workflow.parameters.acceptable-accuracy}}")
    learning_rate_env = k8s.V1EnvVar(
        name="LEARNING_RATE", value="{{workflow.parameters.learning-rate}}")
    epochs_env = k8s.V1EnvVar(
        name="EPOCHS", value="{{workflow.parameters.epochs}}")
    batch_size_env = k8s.V1EnvVar(
        name="BATCH_SIZE", value="{{workflow.parameters.batch-size}}")
    test_amount_env = k8s.V1EnvVar(
        name="TEST_AMOUNT_AMOUNT", value="{{workflow.parameters.test-amount}}")
    requests_delay_env = k8s.V1EnvVar(
        name="REQUESTS_DELAY", value="{{workflow.parameters.requests-delay}}")

    # 1. Download MNIST data
    download = dsl.ContainerOp(
        name="download",
        image="tidylobster/mnist-pipeline-download:latest",       # <-- Replace with correct docker image
        file_outputs={"data_path": "/data_path.txt"})
    download.add_volume(storage_volume)
    download.add_volume_mount(storage_volume_mount)
    download.add_env_variable(mount_path_env)

    # 2. Train and save a MNIST classifier using Tensorflow
    train = dsl.ContainerOp(
        name="train",
        image="tidylobster/mnist-pipeline-train:latest",        # <-- Replace with correct docker image
        file_outputs={"accuracy": "/accuracy.txt"},
        arguments=[download.outputs["data_path"]])

    train.after(download)
    train.set_memory_request('1G')
    train.set_cpu_request('1')

    train.add_volume(storage_volume)
    train.add_volume_mount(storage_volume_mount)
    train.add_env_variable(mount_path_env)
    train.add_env_variable(learning_rate_env)
    train.add_env_variable(epochs_env)
    train.add_env_variable(batch_size_env)

    # 3. Upload trained model to the cluster
    upload = dsl.ContainerOp(
        name="upload",
        image="tidylobster/mnist-pipeline-upload:latest",         # <-- Replace with correct docker image
        file_outputs={"model-version": "/model-version.txt"},
        arguments=[train.outputs["accuracy"], download.outputs["data_path"]])
    upload.after(train)
    
    upload.add_volume(storage_volume) 
    upload.add_volume_mount(storage_volume_mount)
    upload.add_env_variable(mount_path_env)
    upload.add_env_variable(model_name_env)
    upload.add_env_variable(hydrosphere_address_env)
    upload.add_env_variable(learning_rate_env)
    upload.add_env_variable(epochs_env)
    upload.add_env_variable(batch_size_env)

    # 4. Pre-deploy application
    predeploy = dsl.ContainerOp(
        name="predeploy",
        image="tidylobster/mnist-pipeline-predeploy:latest",        # <-- Replace with correct docker image
        arguments=[upload.outputs["model-version"]],
        file_outputs={"predeploy-app-name": "/predeploy-app-name.txt"})
    predeploy.after(upload)

    predeploy.add_env_variable(hydrosphere_address_env)
    predeploy.add_env_variable(application_name_env)
    predeploy.add_env_variable(model_name_env)
    
    # 5. Test the model 
    test = dsl.ContainerOp(
        name="test",
        image="tidylobster/mnist-pipeline-test:latest",               # <-- Replace with correct docker image
        arguments=[predeploy.outputs["predeploy-app-name"], download.outputs["data_path"]])
    test.set_retry(3)
    test.after(predeploy)

    test.add_volume(storage_volume) 
    test.add_volume_mount(storage_volume_mount)
    test.add_env_variable(mount_path_env)
    test.add_env_variable(hydrosphere_address_env)
    test.add_env_variable(application_name_env)
    test.add_env_variable(test_amount_env)
    test.add_env_variable(acceptable_accuracy_env)
    test.add_env_variable(requests_delay_env)

    # 7. Deploy application
    deploy = dsl.ContainerOp(
        name="deploy",
        image="tidylobster/mnist-pipeline-deploy:latest",              # <-- Replace with correct docker image
        arguments=[upload.outputs["model-version"]])
    deploy.after(test)

    deploy.add_env_variable(hydrosphere_address_env)
    deploy.add_env_variable(application_name_env)
    deploy.add_env_variable(model_name_env)
    

if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys

    namespace = "cc2645d0"
    assert namespace is not None, "Assign a namespace variable"

    compiler.Compiler().compile(pipeline_definition, "pipeline.tar.gz")

    untar = "tar -xvf pipeline.tar.gz"
    replace = f"sed -i '' s/minio-service.kubeflow/minio-service.{namespace}/g pipeline.yaml"

    process = subprocess.run(untar.split())
    process = subprocess.run(replace.split())