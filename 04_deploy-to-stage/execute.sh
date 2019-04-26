#!/bin/bash

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
      --model-version)
      MODEL_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
      --model-name)
      MODEL_NAME="$2"
      shift # past argument
      shift # past value
      ;;
      --hydrosphere-address)
      HYDROSPHERE_ADDRESS="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

echo MODEL NAME           = $MODEL_NAME
echo MODEL VERSION        = $MODEL_VERSION
echo HYDROSPHERE ADDRESS  = $HYDROSPHERE_ADDRESS

# Define where Hydrosphere Serving instance is running.
hs cluster add --name serving --server $HYDROSPHERE_ADDRESS
hs cluster use serving

# Retrieve the model version and generate random postfix.
# export POSTFIX=$(head /dev/urandom | LC_ALL=C tr -dc A-Za-z0-9 | head -c 13 ; echo '')
export POSTFIX=stage
export APPLICATION_NAME=${MODEL_NAME}-${POSTFIX}-app
echo ${APPLICATION_NAME} > /stage-app-name.txt

# Create an application manifest.
cat > app.yaml << EOL
kind: Application
name: ${APPLICATION_NAME}

singular:
  model: ${MODEL_NAME}:${MODEL_VERSION}
EOL

# Deploy endpoint application on the Hydrosphere Serving instance.
hs apply -f app.yaml