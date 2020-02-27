FROM htcondor/htmap-exec:v0.5.0

USER root

RUN conda install -y opencv scikit-learn scikit-image networkx \
 && conda clean -y --all

ARG CACHEBUST=1

RUN pip install --no-cache-dir git+https://github.com/JoshKarpel/fish.git

USER htmap
