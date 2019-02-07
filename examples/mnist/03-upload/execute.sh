hs cluster add --name $CLUSTER_NAME --server $CLUSTER_ADDRESS
hs cluster use $CLUSTER_NAME
hs upload --name $MODEL_NAME | tail -n 1 | jq ".version" > /model_version.txt