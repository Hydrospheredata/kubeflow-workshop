FROM python:3.6-slim
RUN pip install tensorflow==1.13.1 requests==2.21.0 numpy==1.14.3 hydro-serving-grpc==2.0.0
ADD ./reqstore.py ./example.db ./sample.py /src/
WORKDIR /src/
ENTRYPOINT [ "python", "sample.py" ]
