FROM python:3.6-slim
RUN apt update && apt install -y jq
RUN pip install hs --pre
ADD ./execute.sh /src/
WORKDIR /src/
ENTRYPOINT [ "bash", "execute.sh" ]
