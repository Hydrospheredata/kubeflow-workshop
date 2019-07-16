# Hydrosphere + Kubeflow Pipelines 

This repository shows how to orchestrate a machine learning workflow with [Kubeflow](https://www.kubeflow.org/) and [Hydrosphere Serving](https://hydrosphere.io/serving/).

## Prerequisites

- Admin access to a Kubernetes cluster
- [Kubeflow](https://www.kubeflow.org/docs/started/getting-started/)
- [Hydrosphere Serving](https://hydrosphere.io/serving-docs/installation.html#kubernetes)

_Note: All components of Kubeflow by default will be installed into `kubeflow` namespace._


## Repository Structure

```
├── Dockerfile         - base image, used for building working steps 
├── README.md          - this document
├── bash.sh            - helper script for working with workshop
├── notebooks          - notebooks for running workshop
├── requirements.txt   - requiremets for building steps
├── serverless         - definitions of Lambda functions and Cloud Formation templates for AWS cloud
├── steps              — actual running steps, that will be executed in the pipeline
├── utilities          - utility script to work with the cloud, orhcestrator, etc.
└── workflows          - definitions of the pipelines
```

## Run Workflow
1. Build and publish all steps of the workflow
    ```
    $ ./bash.sh --build-base --build-workers --docker
    ```
1. Compile origin/subsample pipeline
    ```
    $ ./bash.sh --compile-origin
    $ ./bash.sh --compile-subsample
    ```
1. Upload compiled pipeline `pipeline.tar.gz` to Kubeflow using UI.
1. Create an experiment on Kubeflow and start a run. 

## Help and Support
[![Join the chat at https://gitter.im/Hydrospheredata/hydro-serving](https://badges.gitter.im/Hydrospheredata/hydro-serving.svg)](https://gitter.im/Hydrospheredata/hydro-serving?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![](https://img.shields.io/badge/documentation-latest-af1a97.svg)](https://hydrosphere.io/serving-docs/) 
