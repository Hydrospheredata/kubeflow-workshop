# Hydrosphere + Kubeflow Pipelines 

This repository shows how to orchestrate a machine learning workflow with [Kubeflow](https://www.kubeflow.org/) and [Hydrosphere Serving](https://hydrosphere.io/serving/).

## Prerequisites

- Admin access to a Kubernetes cluster
- [Kubeflow](https://www.kubeflow.org/docs/started/getting-started/)
- [Hydrosphere Serving](https://hydrosphere.io/serving-docs/installation.html#kubernetes)

_Note: All components of Kubeflow by default will be installed into `kubeflow` namespace._

## Run a Workflow

1. Build and publish all stages of the workflow 01-07
1. Adjust `pipeline.py` to point to your published images
1. Compile the pipeline
    ```sh 
    $ python workflows/origin.py
    ```

This will create two files for you: `pipeline.yaml`. You can use this file to start a pipeline execution. 

- (Recommended) Kubeflow Pipelines
    - UI
        1. Open Kubeflow UI and upload `pipeline.yaml` with `Upload Workflow` button
        1. Create an experiment and make a run using this pipeline
    - Shell
        1. Execute `python kubeflow_client.py`

- Argo Workflows
    1. Install [argo](https://github.com/argoproj/argo/blob/master/demo.md#1-download-argo)
    1. Submit a workflow
        ```sh
        $ argo submit pipeline.yaml --watch
        ```
