KUBEFLOW ?= ml-pipeline-ui.k8s.hydrosphere.io
EXPERIMENT ?= Default
REGISTRY ?= hydrosphere
TAG ?= v1

all: origin

origin: compile-origin submit-origin clean
subsample: compile-subsample submit-subsample clean

compile-origin:
	python3 workflows/origin.py -t $(TAG) -r $(REGISTRY)
submit-origin:
	python3 utils/kubeflow.py -f origin.tar.gz -e "$(EXPERIMENT)" -k $(KUBEFLOW)

compile-subsample:
	python3 workflows/subsample.py -t $(TAG) -r $(REGISTRY)
submit-subsample:
	python3 utils/kubeflow.py -f subsample.tar.gz -e "$(EXPERIMENT)" -k $(KUBEFLOW)  

release-all-steps:
	@for path in download train-drift-detector train-model release-drift-detector release-model deploy test; do \
		cd steps/$$path && make release; \
		cd ../../; \
	done
release-all-steps-raw:
	@for path in download train-drift-detector train-model release-drift-detector release-model deploy output test; do \
		cd steps/$$path && make release-raw; \
		cd ../../; \
	done

clean: clean-workers
	rm -rf origin.tar.gz subsample.tar.gz
clean-workers:
	@for path in download train-drift-detector train-model release-drift-detector release-model deploy output test; do \
		cd steps/$$path && make clean; \
		cd ../../; \
	done