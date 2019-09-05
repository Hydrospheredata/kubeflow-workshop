IMAGE_NAME = mnist-pipeline-output
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local: 
	@echo Performing local run
	python3 output.py \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-artifacts-path s3://workshop-hydrosphere/mnist/model/mnist/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=9fac7a9d6c0a3a95745aeb51f074340d \
		--drift-detector-artifacts-path s3://workshop-hydrosphere/mnist/model/mnist-drift-detector/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=0e2faa6685423711827187de42efba7e \
		--model-uri https://dev.k8s.hydrosphere.io/models/37/256/details \
		--drift-detector-uri https://dev.k8s.hydrosphere.io/models/36/255/details \
		--model-application-uri https://dev.k8s.hydrosphere.io/applications/mnist_stage_app \
		--drift-detector-application-uri https://dev.k8s.hydrosphere.io/applications/mnist-drift-detector_app\
		--integration-test-accuracy 0.97 \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-artifacts-path s3://workshop-hydrosphere/mnist/model/mnist/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=9fac7a9d6c0a3a95745aeb51f074340d \
		--drift-detector-artifacts-path s3://workshop-hydrosphere/mnist/model/mnist-drift-detector/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=0e2faa6685423711827187de42efba7e \
		--model-uri https://dev.k8s.hydrosphere.io/models/37/256/details \
		--drift-detector-uri https://dev.k8s.hydrosphere.io/models/36/255/details \
		--model-application-uri https://dev.k8s.hydrosphere.io/applications/mnist_stage_app \
		--drift-detector-application-uri https://dev.k8s.hydrosphere.io/applications/mnist-drift-detector_app\
		--integration-test-accuracy 0.97 \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf cloud.py *.log application_name application_uri 
