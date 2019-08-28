FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install boto3~=1.9.197
RUN pip install tensorflow~=1.13.1
RUN pip install mlflow~=1.0.0

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "train_drift_detector.py" ]
