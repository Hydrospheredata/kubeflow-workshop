FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install boto3~=1.9.197
RUN pip install tensorflow~=1.13.1
RUN pip install scikit-learn~=0.20.2
RUN pip install wo~=0.1.5
RUN pip install mlflow==1.4.0

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "train_model.py" ]
