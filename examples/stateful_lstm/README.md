# HydroServing + Kubeflow + Argo demo

This example shows how to train a model on [Kubeflow](https://www.kubeflow.org/) with [Tensorflow](https://www.tensorflow.org/)
and then to serve it with [HydroServing](https://hydrosphere.io/ml-lambda/)

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

Deploy Argo and Tensorflow Operator:

```
ks init hydro-serving-kubeflow-demo
cd hydro-serving-kubeflow-demo

ks registry add kubeflow github.com/kubeflow/kubeflow/tree/v0.2.5/kubeflow

ks pkg install kubeflow/core@v0.2.5
ks pkg install kubeflow/argo

kubectl create namespace hydroflow
ks generate core kubeflow-core --name=kubeflow-core --namespace=hydroflow
ks generate argo kubeflow-argo --name=kubeflow-argo --namespace=hydroflow

ks apply default -c kubeflow-core
ks apply default -c kubeflow-argo

cd ..
```

Deploy HydroServing platform:

```
helm repo add hydro-serving https://hydrospheredata.github.io/hydro-serving-helm/

helm install --name kubeflow --namespace hydroflow hydro-serving/serving 
```

More details and configurations docs are [here](https://github.com/Hydrospheredata/hydro-serving-helm)

Create Secret with AWS credentials:

```
export AWS_ACCESS_KEY_ID=<aws-key-id>
export AWS_SECRET_ACCESS_KEY=<aws-secret-access-key>

kubectl create secret generic aws-creds --from-literal=awsAccessKeyID=${AWS_ACCESS_KEY_ID} \
  --from-literal=awsSecretAccessKey=${AWS_SECRET_ACCESS_KEY} --namespace hydroflow
```

Create a user (if RBAC is enabled):

```
kubectl apply -f service-account.yaml --namespace hydroflow
```

## Run the workflow

To run the workflow:
```
argo submit model-workflow.yaml \
  --serviceaccount tf-user \
  -n hydroflow \
  -p job-name=job-$(uuidgen  | cut -c -5 | tr '[:upper:]' '[:lower:]') \
  -p namespace=hydroflow \
  -p model-name=stateful_lstm
```

For Argo UI access share the related port:
```
kubectl port-forward deployment/argo-ui 8001:8001 --namespace hydroflow
```
The Argo UI should be available now at http://localhost:8001/

For HydroServing UI access forward the related port:
```
kubectl port-forward deployment/kubeflow-sidecar 8080:8080  --namespace hydroflow
```
You should now be able to visit http://localhost:8080/

## Clean Up

To clean workflow pods you shoud detect its unique name:

```
argo list --namespace hydroflow

NAME                   STATUS      AGE   DURATION
hydro-workflow-rpr6f   Succeeded   6m    1m
hydro-workflow-4hfdf   Succeeded   10m   30s
hydro-workflow-dwvvn   Failed      11m   21s
hydro-workflow-npx4l   Succeeded   35m   29s
```

Then you can easily remove workflow:
```
argo delete --namespace hydroflow <workflow-name>
```
