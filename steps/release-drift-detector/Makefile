IMAGE_NAME = mnist-pipeline-release-drift-detector
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest
BUCKET ?= s3://workshop-hydrosphere-mnist
SAMPLE_VERSION ?= sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7
MODEL_VERSION ?= model-version=04d47c1ba6c13372ab5812adb83ce22a

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local: 
	@echo Performing local run
	python3 release_drift_detector.py \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-path $(BUCKET)/model/mnist-drift-detector/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
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
	docker run --rm -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-path $(BUCKET)/model/mnist-drift-detector/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
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
	rm -rf logs
	rm -rf inputs
	rm -rf outputs
	rm -rf learning_rate
	rm -rf batch_size
	rm -rf loss
	rm -rf data_path 
	rm -rf model_name
	rm -rf model_path
	rm -rf model_uri
	rm -rf model_version
	rm -rf steps
	rm -rf git.*
	rm -rf .hs
	