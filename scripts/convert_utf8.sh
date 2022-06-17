#!/bin/sh


for f in **/*.csv; do
    nvim +":set fileencoding=utf-8 | :wq" $f
done
