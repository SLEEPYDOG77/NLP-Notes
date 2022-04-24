# -*- coding: utf-8 -*-
# @Time    : 2022/4/23 14:55
# @Author  : Zhang Jiaqi
# @File    : w1_2helloworld.py
# @Description: C1W1 the hello world of neural networks coursera

import numpy as np
import tensorflow.keras as keras

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

