# Define where Hydrosphere Serving instance is running.
hs cluster add --name $CLUSTER_NAME --server $CLUSTER_ADDRESS
hs cluster use $CLUSTER_NAME

# Retrieve the model version.
export MODEL_VERSION=$1

# Create an application manifest.
cat > app.yaml << EOL
kind: Application
name: ${APPLICATION_NAME}
singular:
  model: ${MODEL_NAME}:${MODEL_VERSION}
  runtime: hydrosphere/serving-runtime-tensorflow:1.7.0-latest
EOL

# Deploy endpoint application on the Hydrosphere Serving instance.
hs apply -f app.yaml