FROM tensorflow/tensorflow:2.12.0-gpu
RUN apt-get update
COPY requirements.txt /requirements.txt
WORKDIR / 
RUN pip install --no-cache-dir -r /requirements.txt
CMD ["/bin/bash"]