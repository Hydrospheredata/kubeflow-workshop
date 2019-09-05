FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install boto3~=1.9.197
RUN pip install numpy==1.16.2
RUN pip install Pillow==5.2.0
RUN pip install wo~=0.1.3.post4

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "download.py" ]
