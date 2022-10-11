# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 20:04
# @Author  : Zhang Jiaqi
# @File    : preprocess.py
# @Description:

import os
from tqdm import tqdm
import music21 as m21

MID_DATASET_PATH = "surname_checked_midis"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

def load_songs_in_mid(mid_dataset_path):
    songs = []

    for path, subdir, files in os.walk(mid_dataset_path):
        for file in tqdm(files[:50]):
            if file[-3:] == "mid":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
        return True

def preprocess():
    songs = load_songs_in_mid(mid_dataset_path=MID_DATASET_PATH)
    print(len(songs))
    print(songs)


if __name__ == "__main__":
    preprocess()