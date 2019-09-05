IMAGE_NAME = mnist-pipeline-train-model
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest

all: test-local prepare build-container clean test-container push-container
release: build-container clean test-container push-container
release-raw: build-container clean push-container

test-local:
	@echo Performing local run
	python3 train_model.py \
		--data-path s3://workshop-hydrosphere/mnist/data/sample-version=13e4e7f62eb6ac60e44c2094a6cd86b7/ \
		--model-path s3://workshop-hydrosphere/mnist/model/ \
		--model-name mnist \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 5 \
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
		--model-name mnist \
		--learning-rate 0.01 \
		--batch-size 64 \
		--epochs 5 \
		--dev 

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf mnist cloud.py *.log logs_path num_classes model_path mlpipeline-metrics.json \
		mlpipeline-ui-metadata.json accuracy loss global_step average_loss
