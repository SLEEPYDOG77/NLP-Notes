# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 15:00
# @Author  : Zhang Jiaqi
# @File    : music21test1.py
# @Description:

from music21 import *

n = note.Note("D#3")
# n.duration.type = "half"
# n.show()

littleMelody = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
littleMelody.write('xml', fp='score/test1.xml')
# littleMelody.show()
# littleMelody.show('midi')


