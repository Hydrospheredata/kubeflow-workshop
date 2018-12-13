FROM python:3.6-slim

ADD ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

ADD ./*.py /src/
ADD ./*.yaml /src/

WORKDIR /src/