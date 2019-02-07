hs cluster add --name $CLUSTER_NAME --server $CLUSTER_ADDRESS
hs cluster use $CLUSTER_NAME

export MODEL_VERSION=$1
cat > app.yaml << EOL
kind: Application
name: ${APPLICATION_NAME}
singular:
  model: ${MODEL_NAME}:${MODEL_VERSION}
  runtime: hydrosphere/serving-runtime-tensorflow:1.7.0-latest
EOL
hs apply -f app.yaml