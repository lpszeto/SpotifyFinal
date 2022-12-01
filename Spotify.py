import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import librosa 
import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch, features
import tensorflow as tf

# components = []
# file = converter.parse('./smallSet/ACDC.Highway_to_Hell_K.mid')
# for element in file.recurse():
#     components.append(element)
#     print(element)

# def openMidi(midi_path, remove_drums):
#     mf = midi.MidiFile()
#     mf.open(midi_path)
#     mf.read()
#     mf.close()
#     if (remove_drums):
#         for i in range(len(mf.tracks)):
#             mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
#     return midi.translate.midiFileToStream(mf)

# def list_instruments(midi):
#     partStream = midiFile.parts.stream()
#     print("List of instruments found")
#     for p in partStream:
#         aux = p
#         print(p.partName)

# midiFile = openMidi('./smallSet/ACDC.Highway_to_Hell_K.mid', False)
# list_instruments(midiFile)


# Gets the average note duration
fileParsed = converter.parse('./smallSet/ACDC.Highway_to_Hell_K.mid')
feature = features.jSymbolic.AverageNoteDurationFeature(fileParsed)
f = feature.extract()
print('Average Note Duration')
print(f.vector)

## Gets the Initial Tempo
feature = features.jSymbolic.InitialTempoFeature(fileParsed)
f = feature.extract()
print('Initial Tempo')
print(f.vector)



