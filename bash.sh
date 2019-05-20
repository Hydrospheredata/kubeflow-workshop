# check arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -b|--build-base)
    BUILD_BASE=true;
    shift # past argument
    ;;
    -w|--build-workers)
    BUILD_WORKERS=true;
    shift # past argument
    ;;
    -c|--clean-folders)
    CLEAN_FOLDERS=true;
    shift # past argument
    ;;
    --no-cache)
    cache="--no-cache";
    shift # past argument
    ;;
    -oc|--origin-compile)
    COMPILE_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    -or|--origin-run)
    RUN_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    -sc|--sampling-compile)
    COMPILE_SAMPLING_PIPELINE=true;
    shift # past argument
    ;;
    -sr|--sampling-run)
    RUN_SAMPLING_PIPELINE=true;
    shift # past argument
    ;;
    -n|--namespace)
    NAMESPACE=$2
    shift # past argument
    shift # past value
    ;;
  esac
done

# Add default environment variables
[ -z "$DOCKER_ACCOUNT" ] && DOCKER_ACCOUNT="hydrosphere"
[ -z "$TAG" ] && TAG="latest"
[ -z "$DIRECTORY" ] && DIRECTORY="."

# Build base if specified
if [[ $BUILD_BASE ]]; then
  echo "Building base image"
  docker build -t $DOCKER_ACCOUNT/odsc-workshop-base:$TAG -f baseDockerfile $cache .
  docker push $DOCKER_ACCOUNT/odsc-workshop-base:$TAG
fi 

# Build workers if specified
if [[ $BUILD_WORKERS ]]; then
  echo "Building stage images"
  for path in 01_download 01_sample 02_train-model 02_train-autoencoder 03_release-model 03_release-autoencoder 04_deploy 05_test; do 
    IFS=$'_'; arr=($path); unset IFS;
    TAG=$TAG envsubst '$TAG' < "$path/Dockerfile" > "$path/SubsDockerfile"
    docker build -t $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG \
      -f "$path/SubsDockerfile" $cache $path
    docker push $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG
    rm "$path/SubsDockerfile"
  done
fi 

# Clean folders
if [[ $CLEAN_FOLDERS ]]; then
  echo "Cleaning folders"
  for path in ./*; do
    if [ -d $path ] && [ -e $path/Dockerfile ]; then
      for file in $path/*; do 
        IFS=$'/'; arr_name=($file); unset IFS;
        IFS=$'.'; arr_extension=($file); unset IFS;
        if [[ ${arr_name[2]} != 'Dockerfile' && ${arr_extension[2]} != 'py' ]]; then
          echo "delete" $file
          rm -rf $file
        fi
      done
    fi
  done
fi

# Compile and run origin if needed
if [[ $COMPILE_ORIGIN_PIPELINE ]]; then
  echo "Compiling origin pipeline"
  python3 workflows/origin.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

if [[ $RUN_ORIGIN_PIPELINE ]]; then
  echo "Running origin pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi

# Compile and run sampling if needed
if [[ $COMPILE_SAMPLING_PIPELINE ]]; then
  echo "Compiling sampling pipeline"
  python3 workflows/sampling.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

if [[ $RUN_SAMPLING_PIPELINE ]]; then
  echo "Running sampling pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi