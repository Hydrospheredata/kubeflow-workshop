# Define where Hydrosphere Serving instance is running.
hs cluster add --name serving --server $CLUSTER_ADDRESS
hs cluster use serving

# By default tf.estimator.export_saved_model creates a folder with 
# timestamp name, when the model will be saved. `cd` to that folder.
cd ${MOUNT_PATH}; cd models/
cd $(ls -t | head -n1)

# Get accuracy from the previous step
export ACCURACY=$1
export DATA_PATH=$2

cat > serving.yaml << EOL
kind: Model
name: ${MODEL_NAME}
metadata: 
  learning_rate: "${LEARNING_RATE}"
  epochs: "${EPOCHS}"
  batch_size: "${BATCH_SIZE}"
  accuracy: "${ACCURACY}"
  data: "${DATA_PATH}"
payload:
  - "saved_model.pb"
  - "variables/" 
runtime: "hydrosphere/serving-runtime-tensorflow-1.13.1:latest"
contract: 
  name: predict
  inputs:
    imgs:
      type: float32
      shape: [-1, 28, 28]
      profile: image
  outputs:
    probabilities:
      type: float32
      shape: [-1, 10]
    class_ids: 
      type: int64
      shape: [-1, 1]
    logits:
      type: float32
      shape: [-1, 10]
    classes:
      type: string
      shape: [-1, 1]
monitoring:
  - name: Requests
    kind: CounterMetricSpec
    config:
      interval: 15
  - name: Latency
    kind: LatencyMetricSpec
    config:
      interval: 15
  - name: Autoencoder
    kind: ImageAEMetricSpec
    with-health: true
    config:
      threshold: 0.15
      application: mnist-concept-app
EOL

# i.  Upload the model to Hydrosphere Serving
# ii. Parse the status of the model uploading, retrieve the built 
#     model version and write it to the `/model_version.txt` file. 
hs upload | tail -n 1 | jq ".modelVersion" > /model-version.txt