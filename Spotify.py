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

## GETTING SPOTIFY SONG DATA FROM USER
def loginToSpotify():
    with open(r"credentials.txt") as f:
        [SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET] = f.read().split("\n")
        f.close()
    auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def searchPlaylist(sp):
    playlistLink = input("Please Enter Playlist URL\n")
    playlistDict = sp.playlist(playlistLink)
    return playlistDict



## SP is the api connection (spotify object)
sp = loginToSpotify()
playlistDict = searchPlaylist(sp)

totalSongs = playlistDict['tracks']['total']
song_list = []
song_id = []
artist_id = []

playlistSongsFile = open('playlistSongs.csv', 'w')

for i in range(totalSongs):
    artists = [k["name"] for k in playlistDict['tracks']['items'][i]["track"]["artists"]]
    trackName = playlistDict['tracks']['items'][i]['track']['name']
    artistName = artists[0]
    songID = playlistDict['tracks']['items'][i]['track']['id']
    uri = playlistDict['tracks']['items'][i]['track']['uri']

    print(f"{trackName}, {artistName}, {songID}, {uri}")
    
    print(sp.audio_features(songID)[0])