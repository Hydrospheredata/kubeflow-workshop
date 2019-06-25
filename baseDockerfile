FROM python:3.6-slim

RUN apt update && apt install -y jq

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt