FROM htcondor/htmap-exec:v0.5.0

USER root

RUN conda update -y --all \
 && conda install -y opencv \
 && conda clean -y --all

ARG CACHEBUST=1

RUN pip install --no-cache-dir git+https://github.com/JoshKarpel/fish.git \
 && python -c "import fish"

USER htmap
