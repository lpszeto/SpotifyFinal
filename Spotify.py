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

def getAudioFeatures(sp, songID):
    valence = sp.audio_features(songID)[0]['valence']
    acousticness = sp.audio_features(songID)[0]['acousticness']
    danceability = sp.audio_features(songID)[0]['danceability']
    durationMS = sp.audio_features(songID)[0]['duration_ms']
    energy = sp.audio_features(songID)[0]['energy']
    id = sp.audio_features(songID)[0]['id']
    instrumentalness = sp.audio_features(songID)[0]['instrumentalness']
    key = sp.audio_features(songID)[0]['key']
    liveness = sp.audio_features(songID)[0]['liveness']
    loudness = sp.audio_features(songID)[0]['loudness']
    mode = sp.audio_features(songID)[0]['mode']
    speechiness = sp.audio_features(songID)[0]['speechiness']
    tempo = sp.audio_features(songID)[0]['tempo']

    return valence, acousticness, danceability, durationMS, energy, id, instrumentalness, key, liveness, loudness, mode, speechiness, tempo




## SP is the api connection (spotify object)
sp = loginToSpotify()
playlistDict = searchPlaylist(sp)

totalSongs = playlistDict['tracks']['total']
song_list = []
song_id = []
artist_id = []

playlistSongsFile = open('playlistSongs.csv', 'w')
playlistSongsFile.write('valence,year,acousticness,artists,danceability,duration_ms,energy,explicit,id,instrumentalness,key,' \
                                                    'liveness,loudness,mode,name,popularity,release_date,speechiness,tempo\n')

for i in range(totalSongs):
    ## GETS TRACK INFO AND SONG INFO
    artists = [k["name"] for k in playlistDict['tracks']['items'][i]["track"]["artists"]]
    trackName = playlistDict['tracks']['items'][i]['track']['name']
    trackName = trackName.replace(',', "")
    artistName = artists[0]
    songID = playlistDict['tracks']['items'][i]['track']['id']
    uri = playlistDict['tracks']['items'][i]['track']['uri']
    releaseDate = playlistDict['tracks']['items'][i]['track']['album']['release_date']
    releaseDate = releaseDate.split('-')[0]
    explicit = 1 if playlistDict['tracks']['items'][i]['track']['explicit'] == 'True' else 0
    popularity = playlistDict['tracks']['items'][i]['track']['popularity']

    print(f"{trackName}, {artistName}, {songID}, {uri}")
    valence, acousticness, danceability, durationMS, energy, id, instrumentalness, key, liveness, loudness, mode, speechiness, tempo = getAudioFeatures(sp, songID)
    
    playlistSongsFile.write(f'{valence},{releaseDate},{acousticness},{artists},{danceability},{durationMS},{energy},' \
                            f'{explicit},{id},{instrumentalness},{key},{liveness},{loudness},{mode},"{trackName}",{popularity},{releaseDate},{speechiness},{tempo}\n')

playlistSongsFile.close()


# data = pd.read_csv("playlistSongs.csv")

# #K-Means 
# cluster_pipline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
# X = data.select_dtypes(np.number)
# num_col = list(X.columns)
# cluster_pipline.fit(X)
# cluster_labels = cluster_pipline.predict(X)
# data['cluster_label'] = cluster_labels

# #Visualizing Cluster PCA 
# pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
# song_embedding = pca_pipeline.fit_transform(X)
# projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
# projection['title'] = data['name']
# projection['cluster'] = data['cluster_label']


# fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
# fig.show()