FROM python:3.6-slim
RUN pip install numpy==1.14.3 Pillow==5.2.0
ADD ./download.py /src/
WORKDIR /src/
ENTRYPOINT [ "python", "download.py" ]
