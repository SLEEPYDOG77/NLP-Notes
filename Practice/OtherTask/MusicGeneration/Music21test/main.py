# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 19:22
# @Author  : Zhang Jiaqi
# @File    : main.py
# @Description: https://www.zhihu.com/collection/788906936
import notebook.notebookapp
import numpy as np
import os
import random

from keras.layers import *
from keras.models import *
import keras.backend as K

from music21 import *

def read_midi(file):
    """
    读取MIDI文件
    :param file:
    :return: 音乐文件中的一组音符和和弦
    """
    notes = []
    notes_to_parse = None

    print('file:\t', file)
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)
    print('parts:\t', parts)

    if parts is not None:
        for part in parts:
            if 'Piano' in str(part):
                notes_to_parse = part.recurse()
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
    else:
        print('no instrument in this midi.')
    return notes

if __name__ == "__main__":
    files = [i for i in os.listdir('midi') if i.endswith(".mid")]

    for i in files:
        notes = read_midi('midi/' + i)
        print(notes)
        # all_notes.append(read_midi(i))

    # notes = [element for notes in all_notes for element in notes]
    # print(notes)
