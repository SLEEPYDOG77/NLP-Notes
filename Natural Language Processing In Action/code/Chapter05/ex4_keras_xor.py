# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2022/2/26 19:52
# software:PyCharm

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))

model.add(Activation('sigmoid'))
model.summary()

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
x_predict = model.predict(x_train)
print(x_predict)
