FROM python:3.6-slim
RUN pip install scikit-learn==0.20.0 requests==2.21.0 numpy==1.14.3 pandas==0.23.3 tqdm==4.23.4
ADD ./test.py /src/
WORKDIR /src/
ENTRYPOINT [ "python", "test.py" ]
