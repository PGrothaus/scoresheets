import random
import numpy as np
from datasets.emnist import (
    load,
    CHESS_CHARS,
    Example
)

import tensorflow.keras as keras


label_to_id_map = {lbl: i for i, lbl in enumerate(CHESS_CHARS)}
num_classes = len(label_to_id_map)


def prep_data(key):
    train_data = load(key)
    examples = []
    for k, v in train_data.items():
        examples.extend(v)

    random.shuffle(examples)
    imgs = np.concatenate([ex.image for ex in examples], axis=0)
    imgs = imgs[:, :, :, np.newaxis]
    targets = np.asarray([label_to_id_map[ex.label] for ex in examples])
    targets = keras.utils.to_categorical(targets)
    print(key)
    print(imgs.shape)
    print(targets.shape)
    del examples
    return imgs, targets


imgs_train, targets_train = prep_data('train')
imgs_test, targets_test = prep_data('test')

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=56, kernel_size=(5, 5), strides=2, activation='relu',
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(
    3, 3), strides=2, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
model.summary()


data_generator = keras.preprocessing.image.ImageDataGenerator(
    validation_split=.2)
## consider using this for more variety
#data_generator = keras.preprocessing.image.ImageDataGenerator(
#    validation_split=.2, width_shift_range=.2,
#    height_shift_range=.2, rotation_range=60,
#    zoom_range=.2, shear_range=.3)


training_data_generator = data_generator.flow(
    imgs_train, targets_train, subset='training')
validation_data_generator = data_generator.flow(
    imgs_train, targets_train, subset='validation')
history = model.fit_generator(training_data_generator,
                              steps_per_epoch=500, epochs=20,
                              validation_data=validation_data_generator)
