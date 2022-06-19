#!/usr/bin/sh

WORKDIR=/home/jovyan/work

docker run -it --rm --name air_notebook  --security-opt label=disable -v "$(pwd)/BD_Prj.ipynb:${WORKDIR}/BD_Prj.ipynb:rw" -p 8888:8888  jupyter/pyspark-notebook:latest
