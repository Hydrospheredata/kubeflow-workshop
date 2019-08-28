IMAGE_NAME = mnist-pipeline-download
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest

all: prepare test-local prepare build-container clean test-container push-container
release: prepare build-container clean test-container push-container
release-raw: prepare build-container clean push-container

prepare:
	@echo Copying local dependencies
	cp ../../utils/cloud.py .

test-local: prepare
	@echo Performing local run
	python3 download.py \
		--output-data-path s3://workshop-hydrosphere/mnist/data \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--output-data-path s3://workshop-hydrosphere/mnist/data \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf data cloud.py *.log logs_path output_data_path
