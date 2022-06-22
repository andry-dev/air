import neuralprophet
from neuralprophet import NeuralProphet
import streamlit as st
import pandas as pd
import numpy as np

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import folium
import streamlit_folium

import pickle

SEED = 413

neuralprophet.set_random_seed(SEED)


def compute_clusters(df, n_clusters, features=['Latitude', 'Longitude'],
                     prediction_col='cluster', max_iterations=10):

    # SparkML expects columns to an instance of Linalg.Vector
    assembler = VectorAssembler().setInputCols(features).setOutputCol('features')
    assembled_df = assembler.transform(df)

    # Create KMeans instance
    kmeans = pyspark.ml.clustering.KMeans(k=n_clusters, featuresCol='features',
                                          predictionCol=prediction_col)

    # Set kmeans seed
    kmeans.setSeed(SEED)

    # Set maximum iterations
    kmeans.setMaxIter(max_iterations)
    # kmeans.getMaxIter()

    # Fit the model
    model = kmeans.fit(assembled_df)

    # Output a prediction for the first entry in the dataset
    model.predict(assembled_df.head().features)

    # Output transformed df
    transformed = model.transform(assembled_df)
    return (model, transformed)


def compute_map(df, limit):
    map = folium.Map(location=(41.90039976875394,
                     12.488480489531007), zoom_start=11)

    # Gather accidents' coordinates
    accidents_coordinates = df['Latitude', 'Longitude'].limit(
        limit).collect()
    for row in accidents_coordinates:

        point = [row['Latitude'], row['Longitude']]
        folium.Marker(location=point).add_to(map)

    # Add coordinates popup
    map.add_child(folium.LatLngPopup())

    return map


def assign_random_color(number):
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                  'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    return color_list[number % len(color_list)]


def compute_clustered_map(df, model, limit):
    # Obtain a map centered on Rome with Folium
    map = folium.Map(location=(41.90039976875394,
                     12.488480489531007), zoom_start=11,  control_scale=False)
    limit = 27  # @param {type:"slider", min:1, max:10000}

    n_clusters = model.summary.k

    for cluster_number in range(n_clusters):
        filtered_by_cluster = df.filter(
            df['cluster'] == cluster_number)

        # Gather accidents' coordinates
        coordinates_list = filtered_by_cluster['Latitude', 'Longitude'].limit(
            limit).collect()

        # Assign color for the markers
        color = assign_random_color(cluster_number)

        for row in coordinates_list:
            point = [row['Latitude'], row['Longitude']]
            folium.Marker(location=point, icon=folium.Icon(
                color=color)).add_to(map)

        # Add coordinates popup
        map.add_child(folium.LatLngPopup())

    return map


class TimeSerieResult():
    def __init__(self, model, metrics, forecast, df_test):
        self.model = model
        self.metrics = metrics
        self.forecast = forecast
        self.df_test = df_test


def generate_forecast(df, frequency='auto'):
    df = df.select(['ISODate', 'Protocollo']) \
           .groupBy(['ISODate', 'Protocollo']) \
           .count() \
           .groupBy('ISODate') \
           .count() \
           .withColumnRenamed('ISODate', 'ds') \
           .withColumnRenamed('count', 'y') \
           .toPandas()

    events_df = df.copy(deep=True)
    events_df['event'] = 'quarantine'
    events_df = events_df.drop(columns=['y'])
    events_df = events_df[(events_df['ds'] >= '2020-3-9')
                          & (events_df['ds'] < '2020-5-1')]

    # MSE is necessary as RMSE is more influenced by outliers (the quarantine event we add), as explained in NeuralProphet docs
    model = NeuralProphet(epochs=120,
                          learning_rate=0.001,
                          loss_func='MSE',
                          daily_seasonality=False,
                          num_hidden_layers=4,
                          d_hidden=32,
                          )
    model.add_country_holidays('IT')
    model.add_events(['quarantine'])
    history_df = model.create_df_with_events(df, events_df)

    df_train, df_test = model.split_df(
        history_df, valid_p=1.0/12, freq=frequency)
    metrics = model.fit(df_train, validation_df=df_test,
                        progress='plot', freq=frequency)

    future = model.make_future_dataframe(df=history_df,
                                         events_df=events_df,
                                         periods=365,
                                         n_historic_predictions=True)
    forecast = model.predict(df=future)

    return TimeSerieResult(model=model, metrics=metrics, forecast=forecast, df_test=df_test)


st.set_page_config(layout='wide')

spark = SparkSession.builder.appName('AIR').getOrCreate()

st.title('Accidents in Rome')


df = spark.read.csv("dataset.csv",
                    sep=",",
                    header=True,
                    inferSchema=True)
df = df.drop('Unnamed: 37')

st.header('Clustering')


n_clusters = st.slider('Number of clusters',
                       min_value=10, max_value=200, value=70)
accidents_per_cluster = st.slider(
    'Max number of accidents per cluster to show', min_value=1, max_value=100, value=5)

clustering_model, clustered_df = compute_clusters(df, n_clusters=n_clusters)

before_clustering_col, after_clustering_col = st.columns(2)

with before_clustering_col:
    st.subheader('Before clustering')
    streamlit_folium.st_folium(compute_map(
        df, accidents_per_cluster * n_clusters), key='standard_map', width=600, height=600)

with after_clustering_col:
    st.subheader('After clustering')
    clusters_map = compute_clustered_map(
        clustered_df, clustering_model, limit=accidents_per_cluster)

    streamlit_folium.st_folium(
        clusters_map, key='clusters_map', width=600, height=600)


st.header('Time serie')

accidents_col, injured_col, deadly_col = st.columns(3)

with accidents_col:
    accidents_ts = None

    with open('accidents_ts.np', 'rb') as f:
        accidents_ts = pickle.load(f)

    st.subheader('Accidents')
    st.write(accidents_ts.model.plot(accidents_ts.forecast))
    st.write(accidents_ts.model.plot_parameters())

with injured_col:
    injured_ts = None

    with open('injured_ts.np', 'rb') as f:
        injured_ts = pickle.load(f)

    st.subheader(
        'Injured')
    st.write(injured_ts.model.plot(injured_ts.forecast))
    st.write(injured_ts.model.plot_parameters())

with deadly_col:
    deadly_ts = None

    with open('deadly_ts.np', 'rb') as f:
        deadly_ts = pickle.load(f)

    st.subheader('Deadly')
    st.write(deadly_ts.model.plot(deadly_ts.forecast))
    st.write(deadly_ts.model.plot_parameters())
