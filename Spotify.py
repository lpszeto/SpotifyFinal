import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import librosa 
import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch

#test
file = converter.parse('./smallSet/ACDC.Highway_to_Hell_K.mid')

def openMidi(midi_path, remove_drums):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return midi.translate.midiFileToStream(mf)


midiFile = open_midi('./smallSet/ACDC.Highway_to_Hell_K.mid', True)
components = []
for element in file.recurse():
    components.append(element)
    print(element)

partStream = midiFile.parts.stream()
print("List of instruments found")
for p in partStream:
    aux = p
print(p.partName)
