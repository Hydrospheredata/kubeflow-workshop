FROM hydrosphere/odsc-workshop-base:$TAG

ADD ./train.py /src/

WORKDIR /src/
ENTRYPOINT [ "python", "train.py" ]
