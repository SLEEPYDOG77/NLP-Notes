# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 13:15
# @Author  : Zhang Jiaqi
# @File    : w1_1cats_vs_dogs.py
# @Description:

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

print(f"total training cat images: {len(os.listdir(train_cats_dir))}")
print(f"total training dog images: {len(os.listdir(train_dogs_dir))}")
print(f"total validation cat images: {len(os.listdir(validation_cats_dir))}")
print(f"total validation dog images: {len(os.listdir(validation_dogs_dir))}")

ncols = 4
nrows = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fname[pic_index-8: pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fname[pic_index-8: pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis("off")

    img = mpimg.imread(img_path)
    plt.imshow(img)
# plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

import numpy as np
from keras.preprocessing import image

for i in range(1, 9):
    path = os.path.join(f'data/predict/predict{i}.jpg')
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0:
        print(f"{path} is a dog.")
    else:
        print(f"{path} is a cat.")
