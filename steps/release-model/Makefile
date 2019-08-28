IMAGE_NAME = mnist-pipeline-release-model
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
	python3 release_model.py \
		--drift-detector-app mnist-drift-detector \
		--model-name mnist \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-path s3://workshop-hydrosphere/mnist/model/mnist/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=b20dd8dbdee1deae0dd0704f2238df27 \
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
	docker run -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--drift-detector-app mnist-drift-detector \
		--model-name mnist \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-path s3://workshop-hydrosphere/mnist/model/mnist/data-version=13e4e7f62eb6ac60e44c2094a6cd86b7/model-version=b20dd8dbdee1deae0dd0704f2238df27 \
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
	rm -rf data cloud.py *.log data_path model_path model_uri model_version \
		learning_rate epochs git.branch* git.is-dirty accuracy batch_size mnist .hs \
		average_loss global_step loss 
