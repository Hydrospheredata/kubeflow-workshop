FROM hydrosphere/odsc-workshop-base:latest
ADD ./sample.py /src/
WORKDIR /src/

COPY notmnist.npz .
COPY t10k.npz . 
ENTRYPOINT [ "python", "sample.py" ]
