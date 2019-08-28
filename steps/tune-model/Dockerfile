FROM $DOCKER_REGISTRY/$BASE_IMG:$TAG

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "tune_model.py" ]
