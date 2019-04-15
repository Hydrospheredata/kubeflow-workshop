FROM python:3.6-slim
RUN pip install hs
ADD ./execute.sh /src/
WORKDIR /src/
ENTRYPOINT [ "bash", "execute.sh" ]
