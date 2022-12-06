import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import plotly.express as px 
import librosa 
import music21
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from music21 import converter, corpus, instrument, midi, note, chord, pitch, features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# file = open('data.csv', 'w')
# writer = csv.writer(file)

# for song in os.listdir('./smallSet'):
#     # Gets the average note duration
#     fileParsed = converter.parse(os.path.join('./smallSet', song))
#     feature = features.jSymbolic.AverageNoteDurationFeature(fileParsed)
#     f1 = feature.extract()
#     print('Average Note Duration')
#     print(f1.vector)

#     ## Gets the Initial Tempo
#     feature = features.jSymbolic.InitialTempoFeature(fileParsed)
#     f2 = feature.extract()
#     print('Initial Tempo')
#     print(f2.vector)
#     songName = song.split('.mid')
#     songName = songName[0]

#     data = [songName, f1.vector[0], f2.vector[0]]
#     writer.writerow(data)

# file.close()


## GETTING SPOTIFY SONG DATA FROM USER
data = pd.read_csv("data.csv")

#K-Means 
cluster_pipline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data.select_dtypes(np.number)
num_col = list(X.columns)
cluster_pipline.fit(X)
cluster_labels = cluster_pipline.predict(X)
data['cluster_label'] = cluster_labels

#Visualizing Cluster PCA 
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']


fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()