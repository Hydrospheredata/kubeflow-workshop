# Define where Hydrosphere Serving instance is running.
hs cluster add --name serving --server $CLUSTER_ADDRESS
hs cluster use serving

# Retrieve the model version and generate random postfix.
# export APPLICATION_POSTFIX=$(head /dev/urandom | LC_ALL=C tr -dc A-Za-z0-9 | head -c 13 ; echo '')
export APPLICATION_POSTFIX=test
export APPLICATION_NAME=${APPLICATION_NAME}-${APPLICATION_POSTFIX}
echo ${APPLICATION_NAME} > /predeploy-app-name.txt

export MODEL_VERSION=$1

# Create an application manifest.
cat > app.yaml << EOL
kind: Application
name: ${APPLICATION_NAME}

singular:
  model: ${MODEL_NAME}:${MODEL_VERSION}
EOL

# Deploy endpoint application on the Hydrosphere Serving instance.
hs apply -f app.yaml