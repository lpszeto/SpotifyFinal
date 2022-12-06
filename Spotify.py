import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import librosa 
import music21
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from music21 import converter, corpus, instrument, midi, note, chord, pitch, features

file = open('data.csv', 'w')
writer = csv.writer(file)

for song in os.listdir('./smallSet'):
    # Gets the average note duration
    fileParsed = converter.parse(os.path.join('./smallSet', song))
    feature = features.jSymbolic.AverageNoteDurationFeature(fileParsed)
    f1 = feature.extract()
    print('Average Note Duration')
    print(f1.vector)

    ## Gets the Initial Tempo
    feature = features.jSymbolic.InitialTempoFeature(fileParsed)
    f2 = feature.extract()
    print('Initial Tempo')
    print(f2.vector)
    songName = song.split('.mid')
    songName = songName[0]

    data = [songName, f1.vector[0], f2.vector[0]]
    writer.writerow(data)

file.close()


## GETTING SPOTIFY SONG DATA FROM USER


