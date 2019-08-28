IMAGE_NAME = mnist-pipeline-train-drift-detector
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
	python3 train_drift_detector.py \
		--data-path s3://workshop-hydrosphere/mnist/data/sample_version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-path s3://workshop-hydrosphere/mnist/model/ \
		--model-name mnist-drift-detector \
		--learning-rate 0.01 \
		--batch-size 64 \
		--steps 500 \
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
		--model-path s3://workshop-hydrosphere/mnist/model/ \
		--model-name mnist-drift-detector \
		--learning-rate 0.01 \
		--batch-size 64 \
		--steps 500 \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf mnist cloud.py *.log logs_path num_classes model_path mlpipeline-metrics.json loss
