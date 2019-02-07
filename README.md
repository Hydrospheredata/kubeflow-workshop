# Hydrosphere + Kubeflow Pipelines (Argo)

This example shows, how to orchestrate a machine learning workflow with [Kubeflow](https://www.kubeflow.org/) and use [Hydrosphere Serving](https://hydrosphere.io/serving/) for model management. 

## Prerequisites

### Kubernetes Cluster

- Kubernetes 1.9
- AWS S3 Access

### Locally

- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [argo](https://github.com/argoproj/argo/blob/master/demo.md#1-download-argo)
- [ksonnet](https://ksonnet.io/#get-started)
- [helm](https://docs.helm.sh/using_helm/#installing-helm)
- Admin access to kubernetes cluster

## Setting up Kubernetes Cluster

```
mkdir kubeflow
cd kubeflow
export KUBEFLOW_TAG=v0.4.1

curl https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/download.sh | bash

helm repo add hydro-serving https://hydrospheredata.github.io/hydro-serving-helm/

helm install --name serving --namespace kubeflow hydro-serving/serving 
```

More details and configurations docs for serving are [here](https://github.com/Hydrospheredata/hydro-serving-helm).
