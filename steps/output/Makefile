IMAGE_NAME = mnist-pipeline-output
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest
BUCKET = s3://workshop-hydrosphere-mnist
SAMPLE_VERSION = sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7
MODEL_VERSION = model-version=d4dad601fb9cdbb8fac3ab642ee51741

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local: 
	@echo Performing local run
	python3 output.py \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-artifacts-path $(BUCKET)/model/mnist/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--drift-detector-artifacts-path $(BUCKET)/model/mnist-drift-detector/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--model-uri http://localhost/models/37/256/details \
		--drift-detector-uri http://localhost/models/36/255/details \
		--model-application-uri http://localhost/applications/mnist-stage-app \
		--drift-detector-application-uri http://localhost/applications/mnist-drift-detector_app\
		--integration-test-accuracy 0.97 \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run --rm -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-artifacts-path $(BUCKET)/model/mnist/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--drift-detector-artifacts-path $(BUCKET)/model/mnist-drift-detector/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--model-uri http://localhost/models/37/256/details \
		--drift-detector-uri http://localhost/models/36/255/details \
		--model-application-uri http://localhost/applications/mnist-stage-app \
		--drift-detector-application-uri http://localhost/applications/mnist-drift-detector_app\
		--integration-test-accuracy 0.97 \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf outputs
	rm -rf logs
	rm -rf mlpipeline-ui-metadata.json
