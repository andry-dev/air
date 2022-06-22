#!/usr/bin/sh

WORKDIR=/home/jovyan/work

podman run -it --rm --name air_notebook  --security-opt label=disable --ulimit nofile=524288:524288 -v "$(pwd)/models:${WORKDIR}/models:rw" -v "$(pwd)/BD_Prj.ipynb:${WORKDIR}/BD_Prj.ipynb:rw" -p 8888:8888 -p 4050:4050 jupyter/pyspark-notebook
