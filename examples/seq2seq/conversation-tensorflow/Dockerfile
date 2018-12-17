FROM python:3.6
# FROM samuelcolvin/tensorflow-gpu-py36

ADD ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install -y unzip

ADD config/ /src/config/ 
ADD scripts /src/scripts/
ADD seq2seq_attention /src/seq2seq_attention/
ADD ./*.py /src/

WORKDIR /src/