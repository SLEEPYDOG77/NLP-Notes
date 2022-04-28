# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 20:36
# @Author  : Zhang Jiaqi
# @File    : w3_2transfer_learning.py
# @Description: C2W3 Coding Transfer Learning from the Inception Mode

import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

train_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/train'
train_cats_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/train/cats'
train_dogs_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/train/dogs'

validation_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/validation'
validation_cats_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/validation/cats'
validation_dogs_dir = 'F:/Datasets/kaggle/cats_and_dogs_filtered/validation/dogs'

train_cat_fname = os.listdir(train_cats_dir)
print(train_cat_fname[:10])
train_dog_fname = os.listdir(train_dogs_dir)
print(train_dog_fname[:10])

local_weights_file = 'F:/Models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(
    optimizer=RMSprop(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['acc']
)

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_steps=50,
    verbose=2
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(f"acc: {acc}")
print(f"val_acc: {val_acc}")
print(f"loss: {loss}")
print(f"val_loss: {val_loss}")

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.show()




