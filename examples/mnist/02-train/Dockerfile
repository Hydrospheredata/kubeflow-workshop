FROM python:3.6-slim
RUN pip install tensorflow==1.9.0 numpy==1.14.3
ADD ./train.py /src/
WORKDIR /src/
ENTRYPOINT [ "python", "train.py" ]