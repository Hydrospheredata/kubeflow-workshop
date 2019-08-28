FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install boto3~=1.9.197
RUN pip install numpy==1.16.2
RUN pip install requests~=2.22.0

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "test.py" ]
