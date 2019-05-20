FROM hydrosphere/odsc-workshop-base:$TAG

ADD ./release.py /src/

WORKDIR /src/
ENTRYPOINT [ "python", "release.py" ]
