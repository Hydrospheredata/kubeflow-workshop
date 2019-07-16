# check arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --docker) 
    BUILD_DOCKER=true;
    shift # past argument
    ;;
    --aws)
    BUILD_AWS=true;
    shift # past argument
    ;;
    --gcp)
    BUILD_GCP=true;
    shift # past argument
    ;;
    --build-base)
    BUILD_BASE=true;
    shift # past argument
    ;;
    --build-workers)
    BUILD_WORKERS=true;
    shift # past argument
    ;;
    --no-cache)
    cache="--no-cache";
    shift # past argument
    ;;
    --compile-origin)
    COMPILE_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    --run-origin)
    RUN_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    --compile-subsample)
    COMPILE_SUBSAMPLE_PIPELINE=true;
    shift # past argument
    ;;
    --run-subsample)
    RUN_SUBSAMPLE_PIPELINE=true;
    shift # past argument
    ;;
    --deploy)
    DEPLOY=true;
    shift # past argument
    ;;
    --namespace)
    NAMESPACE=$2
    shift # past argument
    shift # past value
    ;;
  esac
done

# Check environment 
if !([ -z ${BUILD_BASE+x} ] || [ -z ${BUILD_WORKERS+x} ]); then
  if [ -z "$BUILD_DOCKER" ] && [ -z "$BUILD_AWS" ]; then
    echo "Either --aws/-a or --docker/-d flags should be passed"
    exit 1
  fi
fi

# Add default environment variables
[ -z "$DOCKER_ACCOUNT" ] && DOCKER_ACCOUNT="hydrosphere"
[ -z "$TAG" ] && TAG="latest"
[ -z "$DIRECTORY" ] && DIRECTORY="."

# Build base if specified
if [[ $BUILD_BASE && $BUILD_DOCKER ]]; then
  echo "Building base image for Docker"
  docker build -t $DOCKER_ACCOUNT/odsc-workshop-base:$TAG -f Dockerfile $cache .
  docker push $DOCKER_ACCOUNT/odsc-workshop-base:$TAG
fi 

# Build workers if specified
if [[ $BUILD_WORKERS && $BUILD_DOCKER ]]; then
  echo "Building stage images for Docker"
  for path in steps/*; do 
    cp utilities/orchestrator.py $path
    cp utilities/storage.py $path
    cp config.env $path
    TAG=$TAG envsubst '$TAG' < "$path/Dockerfile" > "$path/envsubDockerfile"
    docker build -t $DOCKER_ACCOUNT/mnist-pipeline-$(basename $path):$TAG \
      -f "$path/envsubDockerfile" $cache $path
    docker push $DOCKER_ACCOUNT/mnist-pipeline-$(basename $path):$TAG
    rm "$path/envsubDockerfile"
    rm "$path/storage.py"
    rm "$path/orchestrator.py"
    rm "$path/config.env"
  done
fi 

# Package files for AWS Lambda
if [[ $BUILD_WORKERS && $BUILD_AWS ]]; then
  echo "Copying functions for packaging"
  for path in steps/*/; do 
    echo $path
    cp $path/*.py serverless
  done
  if [[ $DEPLOY ]]; then
    echo "Deploying service to AWS"
    cd serverless
    serverless deploy
    rm *.py
  fi
fi 

# Compile origin and subsample piplines
if [[ $COMPILE_ORIGIN_PIPELINE ]]; then
  echo "Compiling origin pipeline"
  if [ ! -z "$NAMESPACE" ]; then
    if [ ! -z "$BUILD_AWS" ]; then
      python3 workflows/origin.py --aws -n $NAMESPACE
    elif [ ! -z "$BUILD_GCP" ]; then
      python3 workflows/origin.py --gcp -n $NAMESPACE
    else 
      python3 workflows/origin.py -n $NAMESPACE
    fi
  else
    if [ ! -z "$BUILD_AWS" ]; then
      python3 workflows/origin.py --aws
    elif [ ! -z "$BUILD_GCP" ]; then
      python3 workflows/origin.py --gcp
    else 
      python3 workflows/origin.py
    fi
  fi
fi

if [[ $COMPILE_SUBSAMPLE_PIPELINE ]]; then
  echo "Compiling subsample pipeline"
  if [ ! -z "$NAMESPACE" ]; then
    if [ ! -z "$BUILD_AWS" ]; then
      python3 workflows/subsample.py --aws -n $NAMESPACE
    elif [ ! -z "$BUILD_GCP" ]; then
      python3 workflows/subsample.py --gcp -n $NAMESPACE
    else 
      python3 workflows/subsample.py -n $NAMESPACE
    fi
  else
    if [ ! -z "$BUILD_AWS" ]; then
      python3 workflows/subsample.py --aws
    elif [ ! -z "$BUILD_GCP" ]; then
      python3 workflows/subsample.py --gcp
    else 
      python3 workflows/subsample.py
    fi
  fi
fi

# Run origin and subsample pipelines
if [[ $RUN_ORIGIN_PIPELINE ]]; then
  echo "Running origin pipeline"
  python3 kubeflow.py -n $NAMESPACE -f pipeline.yaml
fi

if [[ $RUN_SUBSAMPLE_PIPELINE ]]; then
  echo "Running subsample pipeline"
  python3 kubeflow.py -n $NAMESPACE -f pipeline.yaml
fi