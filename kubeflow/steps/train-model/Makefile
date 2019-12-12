IMAGE_NAME = mnist-pipeline-train-model
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest
BUCKET ?= s3://workshop-hydrosphere-mnist
SAMPLE_VERSION ?= sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local:
	@echo Performing local run
	python3 train_model.py \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-path $(BUCKET)/model/ \
		--model-name mnist \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 1 \
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
		--model-path $(BUCKET)/model/ \
		--model-name mnist \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 1 \
		--dev 

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf logs 
	rm -rf inputs
	rm -rf outputs
	rm -rf mlpipeline-metrics.json
	rm -rf mlpipeline-ui-metadata.json
	rm -rf model_path
	rm -rf loss
	rm -rf accuracy
	rm -rf average_loss
	rm -rf accuracy_loss
	rm -rf global_step
	rm -rf num_classes
