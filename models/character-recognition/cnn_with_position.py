import random
import numpy as np
from datasets.character_recognition.move_generator import (
    load_generated_moves,
    GeneratedExample,
)
from datasets.emnist import (
    load,
    plot_example,
    CHESS_CHARS,
    Example
)
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from utils import plot_confusion_matrix


label_to_id_map = {lbl: i for i, lbl in enumerate(CHESS_CHARS)}
num_classes = len(label_to_id_map)


position_map = {"is-first": 0,
                "is-last": 1,
                }


def prep_data(key):
    examples = load_generated_moves(key)

    #train_data = load(key)
    #examples = []
    # for k, v in train_data.items():
#        examples.extend(v)
    plot_example(examples[0])

    random.shuffle(examples)
    print(type(examples[0].image))
    print(examples[0].image.shape)
    imgs = np.concatenate([ex.image for ex in examples], axis=0)
    print(imgs.shape)
    imgs = imgs[:, :, :, np.newaxis]
    targets = np.asarray([label_to_id_map[ex.label] for ex in examples])
    targets = keras.utils.to_categorical(targets)
    positions = [position_map.get(ex.position, 2) for ex in examples]
    positions = keras.utils.to_categorical(positions)
    print(key)
    print(imgs.shape)
    print(targets.shape)
    print(positions.shape)
    del examples
    return imgs, targets, positions


def createGenerator(X, I, Y, shuffle=True):

    while True:
        # suffled indices
        idx = np.random.permutation(X.shape[0])
        # create image generator
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # 180,  # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=10,
            # 0.1,  # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # 0.1,  # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        batches = datagen.flow(X[idx], Y[idx], batch_size=64, shuffle=shuffle)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], I[idx[idx0:idx1]]], batch[1]

            idx0 = idx1
            if idx1 >= X.shape[0]:
                break


images_train, targets_train, positions_train = prep_data('train')
images_test, targets_test, positions_test = prep_data('test')

net = keras.models.Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=56, kernel_size=(5, 5), strides=2, activation='relu',
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(
    3, 3), strides=2, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(2, 2), activation='relu'))
# model.add(keras.layers.Conv2D(
#    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))

extra = keras.model.Sequential()
extra.add(keras.layers.Dense(units=128, activation='relu'))

net.add(keras.layers.core.Merge([model, extra], mode='concat', concat_axis=2))
net.add(keras.layers.Dense(units=num_classes, activation='softmax'))
net.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['categorical_accuracy'])
net.summary()


training_data_generator = createGenerator(images_train, positions_train,
                                       targets_train, shuffle=True)
validation_data_generator = createGenerator(images_test, positions_test,
                                       targets_test, shuffle=True)


#training_data_generator = data_generator_train.flow(
#    imgs_train, targets_train)  # , subset='training')
#validation_data_generator = data_generator_valid.flow(
#    imgs_test, targets_test, shuffle=False)  # , subset='validation')
history = net.fit_generator(training_data_generator,
                              steps_per_epoch=500, epochs=5,
                              validation_data=validation_data_generator)

Y_pred = net.predict_generator(validation_data_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(targets_test, axis=1)
print(y_test.shape, y_pred.shape, Y_pred.shape, targets_test.shape)
print(y_pred.shape)
print(targets_test.shape)
print('Confusion Matrix')
print(confusion_matrix(np.argmax(targets_test, axis=1), y_pred))
print('Classification Report')
target_names = CHESS_CHARS
print(classification_report(np.argmax(targets_test, axis=1),
                            y_pred, target_names=target_names))


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=CHESS_CHARS,
                      title='Confusion matrix, without normalization')

plt.show()
