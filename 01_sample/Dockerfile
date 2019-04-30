FROM tidylobster/odsc-base:1.0
ADD ./sample.py /src/
WORKDIR /src/
RUN curl -Ss -L -o notmnist.tgz https://s3.eu-central-1.amazonaws.com/hydroresearch-outlier-detector/mnist-autoencoder/notmnist.tar.gz && \
    tar -xzvf notmnist.tgz && \
    curl -Ss -L -o /data/data-test.tgz https://s3.eu-central-1.amazonaws.com/hydroresearch-outlier-detector/mnist-autoencoder/mnist-test.tar.gz && \
    tar -xzvf data-test.tgz

ENTRYPOINT [ "python", "sample.py" ]
