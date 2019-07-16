FROM hydrosphere/odsc-workshop-base:$TAG

COPY *.py /src/
COPY config.env /src/
WORKDIR /src/

ENTRYPOINT [ "python", "release_drift_detector.py" ]
