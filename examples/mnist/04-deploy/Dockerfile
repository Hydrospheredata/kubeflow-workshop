FROM python:3.6-slim
RUN pip install hs==0.1.5
ADD ./execute.sh /src/
WORKDIR /src/
ENTRYPOINT [ "bash", "execute.sh" ]
