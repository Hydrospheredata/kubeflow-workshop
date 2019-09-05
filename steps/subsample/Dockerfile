FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install boto3~=1.9.197
RUN pip install numpy~=1.17.0
RUN pip install psycopg2-binary~=2.7.5
RUN pip install requests~=2.22.0
RUN pip install hydro-serving-grpc~=2.1.0rc1
RUN pip install tqdm~=4.23.4 
RUN pip install wo~=0.1.3.post4

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "subsample.py" ]
