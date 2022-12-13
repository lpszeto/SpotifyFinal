import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import plotly.express as px
import librosa
import music21
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from kneed import KneeLocator

## n is the amount of songs you want to recommend
n = 25
data = pd.read_csv("data.csv")

#K-Means 
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data.select_dtypes(np.number)
num_col = list(X.columns)
cluster_pipeline.fit(X)
cluster_labels = cluster_pipeline.predict(X)
data['cluster_label'] = cluster_labels

#Visualizing Cluster PCA 
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['artists'] = data['artists']
projection['cluster'] = data['cluster_label']


fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title', 'artists'])
fig.show()

colOrder = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

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



# SP is the api connection (spotify object)
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

    ## IF THERE ARE MORE THAN ONE ARTISTS PUT THE LIST IN QUOTATIONS SO PANDAS CAN READ IT
    if len(artists) > 1:
        print(f"{trackName}, {artistName}, {songID}, {uri}")
        valence, acousticness, danceability, durationMS, energy, id, instrumentalness, key, liveness, loudness, mode, speechiness, tempo = getAudioFeatures(sp, songID)
        
        playlistSongsFile.write(f'{valence},{releaseDate},{acousticness},"{artists}",{danceability},{durationMS},{energy},' \
                                f'{explicit},{id},{instrumentalness},{key},{liveness},{loudness},{mode},"{trackName}",{popularity},{releaseDate},{speechiness},{tempo}\n')
    else:
        print(f"{trackName}, {artistName}, {songID}, {uri}")
        valence, acousticness, danceability, durationMS, energy, id, instrumentalness, key, liveness, loudness, mode, speechiness, tempo = getAudioFeatures(sp, songID)
        
        playlistSongsFile.write(f'{valence},{releaseDate},{acousticness},{artists},{danceability},{durationMS},{energy},' \
                                f'{explicit},{id},{instrumentalness},{key},{liveness},{loudness},{mode},"{trackName}",{popularity},{releaseDate},{speechiness},{tempo}\n')
    print(f"Retrieved data for song {i+1}/{totalSongs}")
playlistSongsFile.close()


# data = pd.read_csv("playlistSongs.csv", on_bad_lines='skip')
# X = data.select_dtypes(np.number)
# print(X)
# sse = []
# for k in range(1,20):
#     kmeans = KMeans(n_clusters=k).fit(X)
#     X["clusters"] = kmeans.labels_
#     print(X["clusters"])   
#     sse.append(kmeans.inertia_)
# kl = KneeLocator(range(1,20),sse, curve="convex", direction="decreasing")

# #K-Means 
# cluster_pipline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=kl.elbow, verbose=False))], verbose=False)
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
# projection['artists'] = data['artists']
# projection['cluster'] = data['cluster_label']


# fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title', 'artists'])
# fig.show()

## Creates a Vector of the playlists songs the user has chosen and then turns it to an array
songVectorData = []
newData = pd.read_csv("playlistSongs.csv", on_bad_lines="skip", encoding='cp1252')
for i in range(newData.shape[0]):
    songData = newData.loc[i,:]
    songVector = songData[colOrder].values
    songVectorData.append(songVector)
songDataArray = np.array(list(songVectorData))
meanVector = np.mean(songDataArray, axis=0) ## Takes the mean and turns it into one array
print(meanVector)

## CREATE LOOP OF RANDOM 3
scaler = cluster_pipeline.steps[0][1]
scaled_data = scaler.transform(data[colOrder])
scaled_song_center = scaler.transform(meanVector.reshape(1, -1))
distances = cdist(scaled_song_center, scaled_data, 'cosine')
index = list(np.argsort(distances)[:, :n][0])
    
rec_songs = data.iloc[index]
print(rec_songs)

