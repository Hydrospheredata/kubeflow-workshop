# Define where Hydrosphere Serving instance is running.
hs cluster add --name $CLUSTER_NAME --server $CLUSTER_ADDRESS
hs cluster use $CLUSTER_NAME

# By default tf.estimator.export_saved_model creates a folder with 
# timestamp name, when the model will be saved. `cd` to that folder.
cd ${MNIST_MODELS_DIR}
cd $(ls -t | head -n1)

# i.  Upload the model to Hydrosphere Serving
# ii. Parse the status of the model uploading, retrieve the built 
#     model version and write it to the `/model_version.txt` file. 
hs upload --name $MODEL_NAME | tail -n 1 | jq ".version" > /model_version.txt

# Push the training files and make a snapshot of them. 
export MODEL_VERSION=$(cat /model_version.txt)
hs profile push --model-version "$MODEL_NAME:$MODEL_VERSION" $MNIST_DATA_DIR