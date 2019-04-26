!/bin/bash

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

# Create an application manifest.
cat > app.yaml << EOL
kind: Application
name: ${MODEL_NAME}-app

singular:
  model: ${MODEL_NAME}:${MODEL_VERSION}
EOL

# Deploy endpoint application on the Hydrosphere Serving instance.
hs apply -f app.yaml