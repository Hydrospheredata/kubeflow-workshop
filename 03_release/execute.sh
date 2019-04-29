#!/bin/bash

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
      --data-path)
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
      --mount-path)
      MOUNT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
      --model-name)
      MODEL_NAME="$2"
      shift # past argument
      shift # past value
      ;;
      --accuracy)
      ACCURACY="$2"
      shift # past argument
      shift # past value
      ;;
      --hydrosphere-address)
      HYDROSPHERE_ADDRESS="$2"
      shift # past argument
      shift # past value
      ;;
      --learning-rate)
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
      --epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
      --batch-size)
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

echo DATA PATH            = $DATA_PATH
echo MOUNT PATH           = $MOUNT_PATH
echo MODEL NAME           = $MODEL_NAME
echo ACCURACY             = $ACCURACY
echo HYDROSPHERE ADDRESS  = $HYDROSPHERE_ADDRESS
echo LEARNING RATE        = $LEARNING_RATE
echo EPOCHS               = $EPOCHS
echo BATCH SIZE           = $BATCH_SIZE

# Define where Hydrosphere Serving instance is running.
hs cluster add --name serving --server $HYDROSPHERE_ADDRESS
hs cluster use serving

# By default tf.estimator.export_saved_model creates a folder with 
# timestamp name, when the model will be saved. `cd` to that folder.
cd ${MOUNT_PATH}; cd models/
cd $(ls -t | head -n1)

# Define contract for the model
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
  - name: Accuracy
    kind: AccuracyMetricSpec
EOL

# i.  Upload the model to Hydrosphere Serving
# ii. Parse the status of the model uploading, retrieve the built 
#     model version and write it to the `/model_version.txt` file. 
hs upload | tail -n 1 | jq ".modelVersion" > /model-version.txt