# Hydrosphere + Kubeflow Pipelines 

This repository shows how to orchestrate a machine learning workflow with [Kubeflow](https://www.kubeflow.org/) and [Hydrosphere Serving](https://hydrosphere.io/serving/).

## Prerequisites

- Admin access to a Kubernetes cluster
- [Kubeflow](https://www.kubeflow.org/docs/started/getting-started/)
- [Hydrosphere Serving](https://hydrosphere.io/serving-docs/installation.html#kubernetes)

## Repository Structure

```
├── notebooks          - notebooks for running workshop and other required operations
├── steps              - pipeline steps executed in the workflow 
├── utils              - utility scripts to work with the cloud, orchestrator, etc.
└── workflows          - definitions of the pipelines
```

## Operations management
* Build, test and publish all steps of the workflow
    ```
    $ REGISTRY=<your_registry> TAG=<tag_of_the_image> make release-all-steps
    ```
* Build and publish all steps of the workflow without testing
    ```
    $ REGISTRY=<your_registry> TAG=<tag_of_the_image> make release-all-steps-raw
    ```
* Compile and submit origin pipeline for execution
    ```
    $ EXPERIMENT=Default KUBEFLOW=<kubeflow_instance_uri> make origin
    ```

## Help and Support
[![Join the chat at https://gitter.im/Hydrospheredata/hydro-serving](https://badges.gitter.im/Hydrospheredata/hydro-serving.svg)](https://gitter.im/Hydrospheredata/hydro-serving?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![](https://img.shields.io/badge/documentation-latest-af1a97.svg)](https://hydrosphere.io/serving-docs/) 
