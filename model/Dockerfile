FROM tensorflow/tensorflow:1.7.0

WORKDIR /src
ADD model_def.py /src/model_def.py
ADD training.py /src/training.py
ADD export.py /src/export.py

CMD /usr/bin/python training.py && /usr/bin/python export.py
