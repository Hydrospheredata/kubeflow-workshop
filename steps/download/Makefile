IMAGE_NAME = mnist-pipeline-download
DOCKER_REGISTRY ?= hydrosphere
BUCKET ?= s3://workshop-hydrosphere-mnist
TAG ?= latest

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local:
	@echo Performing local run
	python3 download.py \
		--output-data-path $(BUCKET)/data \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run --rm -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--output-data-path $(BUCKET)/data \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf data
	rm -rf outputs
	rm -rf logs
	rm -rf output_data_path
	rm -rf logs_path
