In order to execute pipeline, create additional resources inside the cluster. 

```sh
$ kubectl apply -f pvc.yaml
```

Compile pipeline with: 

```sh 
$ python pipeline.py pipeline.tar.gz && tar -xvf pipeline.tar.gz
```

This will create two additional files for you: `pipeline.yaml` and `pipeline.tar.gz`. You can use both of those files to start pipeline execution. 

- (Recommended) Kubeflow Pipelines
    1. Get an access to Kubeflow Pipelines by: 
        ```sh
        $ kubectl port-forward deployment/ambassador 8085:80
        ```
    1. Open `localhost/pipeline`
    1. Press `Upload Workflow` button and upload `pipeline.tar.gz` file. 
    1. Create a run from that pipeline

- Argo Workflows
    1. Install [argo](https://github.com/argoproj/argo/blob/master/demo.md#1-download-argo)
    1. Submit workflow
        ```sh
        $ argo submit pipeline.yaml --watch
        ```