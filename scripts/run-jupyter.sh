#!/usr/bin/sh

WORKDIR=/home/jovyan/work

docker run -it --rm --name air_notebook --ulimit nofile=524288:524288 --security-opt label=disable -v "$(pwd)/model:${WORKDIR}/model" -v "$(pwd)/datasets/aggregate.csv:${WORKDIR}/aggregate.csv" -v "$(pwd)/BD_Prj.ipynb:${WORKDIR}/BD_Prj.ipynb:rw" -p 8888:8888  jupyter/pyspark-notebook:latest
