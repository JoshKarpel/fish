FROM htcondor/htmap-exec:v0.4.1

USER root

RUN conda install -y opencv \
 && conda clean -y --all

ARG CACHEBUST=1

RUN pip install --no-cache-dir git+https://github.com/JoshKarpel/fish.git

USER htmap
