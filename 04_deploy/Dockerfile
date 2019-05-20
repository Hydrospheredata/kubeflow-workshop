FROM hydrosphere/odsc-workshop-base:$TAG

ADD ./deploy.py /src/

WORKDIR /src/
ENTRYPOINT [ "python", "deploy.py" ]
