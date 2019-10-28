IMAGE_NAME = mnist-pipeline-release-model
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest
BUCKET ?= s3://workshop-hydrosphere-mnist
SAMPLE_VERSION ?= sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7
MODEL_VERSION ?= model-version=d4dad601fb9cdbb8fac3ab642ee51741

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local:
	@echo Performing local run
	python3 release_model.py \
		--drift-detector-app mnist-drift-detector \
		--model-name mnist \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-path $(BUCKET)/model/mnist/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--accuracy 0.9639000296592712 \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 10 \
		--average-loss 0.16841885 \
		--global-step 938 \
		--loss 10.727316 \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run --rm -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--drift-detector-app mnist-drift-detector \
		--model-name mnist \
		--data-path $(BUCKET)/data/$(SAMPLE_VERSION)/ \
		--model-path $(BUCKET)/model/mnist/$(SAMPLE_VERSION)/$(MODEL_VERSION)/ \
		--accuracy 0.9639000296592712 \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 10 \
		--average-loss 0.16841885 \
		--global-step 938 \
		--loss 10.727316 \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf logs
	rm -rf inputs
	rm -rf outputs
	rm -rf git.*
	rm -rf batch_size 
	rm -rf learning_rate
	rm -rf loss
	rm -rf data_path
	rm -rf model_path
	rm -rf model_name
	rm -rf model_uri
	rm -rf model_version
	rm -rf global_step
	rm -rf epochs 
	rm -rf accuracy
	rm -rf average_loss
	rm -rf drift_detector_app
	rm -rf .hs
