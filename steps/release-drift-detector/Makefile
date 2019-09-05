IMAGE_NAME = mnist-pipeline-release-drift-detector
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local: 
	@echo Performing local run
	python3 release_drift_detector.py \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-path s3://workshop-hydrosphere/mnist/model/mnist-drift-detector/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=ecaebfd6b1f36f85ac46c35997ab0c87 \
		--model-name mnist-drift-detector \
		--learning-rate 0.01 \
		--batch-size 64 \
		--steps 500 \
		--loss 0.05402803793549538 \
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
		--model-path s3://workshop-hydrosphere/mnist/model/mnist-drift-detector/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=ecaebfd6b1f36f85ac46c35997ab0c87 \
		--model-name mnist-drift-detector \
		--learning-rate 0.01 \
		--batch-size 64 \
		--steps 500 \
		--loss 0.05402803793549538 \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf data cloud.py *.log logs_path output_data_path git.branch* git.is-dirty \
		learning_rate loss model_path model_uri model_version data_path batch_size steps \
		mnist .hs
