# Run Workflow

In order to execute the pipeline, create additional resources inside the cluster. 

```sh
$ kubectl apply -f pvc.yaml
```

Compile the pipeline with: 

```sh 
$ python pipeline.py pipeline.tar.gz && tar -xvf pipeline.tar.gz
```

This will create two additional files for you: `pipeline.yaml` and `pipeline.tar.gz`. You can use both of these files to start pipeline execution. 

- (Recommended) Kubeflow Pipelines
    1. Open Kubeflow UI and upload `pipeline.tar.gz` with `Upload Workflow` button
    1. Create an experiment and make a run from that pipeline

- Argo Workflows
    1. Install [argo](https://github.com/argoproj/argo/blob/master/demo.md#1-download-argo)
    1. Submit workflow
        ```sh
        $ argo submit pipeline.yaml --watch
        ```