# Define where Hydrosphere Serving instance is running.
hs cluster add --name serving --server $CLUSTER_ADDRESS
hs cluster use serving

# Get predeploy application name
export APPLICATION_NAME=$1

# Remove predeploy application
hs app rm ${APPLICATION_NAME}