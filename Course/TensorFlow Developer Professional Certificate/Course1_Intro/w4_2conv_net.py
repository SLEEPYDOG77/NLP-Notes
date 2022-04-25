# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 15:32
# @Author  : Zhang Jiaqi
# @File    : w4_2conv_net.py
# @Description: C1W4 Walking through developing a ConvNet

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_horse_dir = os.path.join('data/horse-or-human/horses')
train_human_dir = os.path.join('data/horse-or-human/humans')
train_horse_name = os.listdir(train_horse_dir)
train_human_name = os.listdir(train_human_dir)

# print(train_horse_name[:10])
# print(train_human_name[:10])

# print(f"total traning horse images: {len(os.listdir(train_horse_dir))}")
# print(f"total traning human images: {len(os.listdir(train_human_dir))}")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_name[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_name[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("off")

    img = mpimg.imread(img_path)
    plt.imshow(img)
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

train_dir = 'data/horse-or-human'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# validation_dir = 'data/horse-or-human'
# test_dategen = ImageDataGenerator(rescale=1./255)
# validation_generator = test_dategen.flow_from_directory(
#     validation_dir,
#     target_size=(300, 300),
#     batch_size=128,
#     class_mode='binary'
# )

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    # validation_data=validation_generator,
    # validation_steps=8,
    verbose=1
)


for i in range(1, 8):
    path = os.path.join(f'data/predict/predict{i}.jpg')
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(f"{path} is a human.")
    else:
        print(f"{path} is a horse.")


