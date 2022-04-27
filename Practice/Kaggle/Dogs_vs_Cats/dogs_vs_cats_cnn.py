# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 19:26
# @Author  : Zhang Jiaqi
# @File    : dogs_vs_cats_cnn.py
# @Description:

import os
import random
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

for dirname, _, filenames in os.walk('F:/Datasets/kaggle/dogs-vs-cats'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

work_path = 'F:/Datasets/kaggle/dogs-vs-cats-filtered'
# os.mkdir(work_path)

# local_zip = 'F:/Datasets/kaggle/dogs-vs-cats/train.zip'
# # unzip
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall(work_path)
#
# local_zip = 'F:/Datasets/kaggle/dogs-vs-cats/test1.zip'
# # unzip
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall(work_path)
#
# zip_ref.close()

train_path = os.path.join(work_path, 'train')
test_path = os.path.join(work_path, 'test1')

# extract labels
train_df = pd.DataFrame({
    'image_name': os.listdir(train_path)
})
train_df['label'] = train_df['image_name'].apply(lambda x: x.split('.')[0])
print(train_df[:5])

# test_df = pd.DataFrame({
#     'image_name': os.listdir(test_path)
# })
# test_df['label'] = test_df['image_name'].apply(lambda x: x.split('.')[0])
# print(test_df[:5])

dog_path_train = os.path.join(train_path, 'dog')
# os.mkdir(dog_path_train)
# dog_df_train = train_df[train_df.label=='dog']
# for n in tqdm(dog_df_train.image_name):
#     os.rename((os.path.join(train_path, n)), (os.path.join(dog_path_train, n)))

cat_path_train = os.path.join(train_path, 'cat')
# os.mkdir(cat_path_train)
# cat_df_train = train_df[train_df.label=='cat']
# for n in tqdm(cat_df_train.image_name):
#     os.rename((os.path.join(train_path, n)), (os.path.join(cat_path_train, n)))

base_dir = 'F:/Datasets/kaggle/dogs-vs-cats-filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test1')

train_cats_dir = os.path.join(train_dir, 'cat')
train_dogs_dir = os.path.join(train_dir, 'dog')
train_cats_names = os.listdir(train_cats_dir)
train_dogs_names = os.listdir(train_dogs_dir)

print(f"num of cats images in training set: {len(train_cats_names)}")
print(f"num of dogs images in training set: {len(train_dogs_names)}")

print(f"num of images in validation set: {len(os.listdir(validation_dir))}")

# nrows = 4
# ncols = 4
# pic_index = 0
#
# fig = plt.gcf()
# fig.set_size_inches(nrows * 4, ncols * 4)
#
# next_cat_pic = [os.path.join(train_cats_dir, fname) for fname in train_cats_names[pic_index:pic_index + 8]]
# next_dog_pic = [os.path.join(train_dogs_dir, fname) for fname in train_dogs_names[pic_index:pic_index + 8]]
#
# for i, img_path in enumerate(next_cat_pic + next_dog_pic):
#     sp = plt.subplot(nrows, ncols, i + 1)
#     sp.axis('off')
#
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
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
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

# acc = history.history['acc']
# # val_acc = history.history['val_acc']
# loss = history.history['loss']
# # val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc)
# # plt.plot(epochs, val_acc)
# plt.title('Training accuracy')
# plt.show()
#
# plt.plot(epochs, loss)
# # plt.plot(epochs, val_loss)
# plt.title('Training Loss')
# plt.show()


sub_filepath = 'F:/Datasets/kaggle/dogs-vs-cats/sampleSubmission.csv'
sub_df = pd.read_csv(sub_filepath, index_col=False)
print(sub_df)

base_dir = 'F:\Datasets\kaggle\dogs-vs-cats-filtered'
validation_dir = os.path.join(base_dir, 'test1')

import numpy as np
from keras.preprocessing import image

for index in range(len(sub_df['label'])):
    img_path = os.path.join(validation_dir, f"{sub_df['id'][index]}.jpg")
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0:
        print(f"{img_path} is a dog.")
        sub_df['label'][index] = classes
    else:
        print(f"{img_path} is a cat.")
        sub_df['label'][index] = classes

output_file = 'F:/Datasets/kaggle/dogs-vs-cats/output.csv'
sub_df.to_csv(output_file)

