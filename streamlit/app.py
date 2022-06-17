import streamlit as st
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('CirnoWithMustache').getOrCreate()

st.title('CirnoWithMustache')

df = spark.createDataFrame(data=[
    ('a', 10),
    ('b', 50),
    ('c', 2)
], schema=['col1', 'col2'])


pdf = df.toPandas()

st.write(pdf)
