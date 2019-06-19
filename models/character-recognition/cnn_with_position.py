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

BATCH_SIZE = 300


def prep_data(key):
    examples = load_generated_moves(key)

    # train_data = load(key)
    # examples = []
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
        if shuffle:
            print("shuffle")
            idx = np.random.permutation(X.shape[0])
        else:
            print("do not shuffle")
            idx = np.asarray(list(range(X.shape[0])))
        # create image generator
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # 180,  # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=10 if shuffle else 0,
            # 0.1,  # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1 if shuffle else 0.,
            # 0.1,  # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1 if shuffle else 0.,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        batches = datagen.flow(
            X[idx], Y[idx], batch_size=BATCH_SIZE, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], I[idx[idx0:idx1]]], batch[1]

            idx0 = idx1
            if idx1 >= X.shape[0]:
                break


images_train, targets_train, positions_train = prep_data('train')
images_test, targets_test, positions_test = prep_data('test')


# merge samples, two input must be same shape
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=56, kernel_size=(5, 5),
                              strides=2, activation='relu',
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(
    3, 3), strides=2, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Conv2D(
    filters=128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))

extra = keras.models.Sequential()
extra.add(keras.layers.Dense(input_shape=(3,), units=128, activation='relu'))
conc = keras.layers.multiply(inputs=[model.output, extra.output])
out = keras.layers.Dense(3*num_classes, activation='relu')(conc)
out = keras.layers.Dense(num_classes, activation='relu')(out)

net = keras.models.Model([model.input, extra.input], out)

net.compile(loss='categorical_crossentropy',
            optimizer='Adam', metrics=['categorical_accuracy'])
net.summary()


training_data_generator = createGenerator(images_train, positions_train,
                                          targets_train, shuffle=True)
validation_data_generator = createGenerator(images_test, positions_test,
                                            targets_test, shuffle=False)
predict_data_generator = createGenerator(images_test, positions_test,
                                            targets_test, shuffle=False)


n_steps = targets_test.shape[0] // BATCH_SIZE
history = net.fit_generator(training_data_generator,
                            steps_per_epoch=500, epochs=5,
                            validation_data=validation_data_generator,
                            validation_steps=n_steps,
                            )

loss = net.evaluate_generator(validation_data_generator,
                               steps=n_steps)
Y_pred = net.predict_generator(predict_data_generator,
                               steps=n_steps)
print(loss)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(targets_test, axis=1)
print(y_test.shape, y_pred.shape, Y_pred.shape, targets_test.shape)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = CHESS_CHARS
print(classification_report(y_test,
                            y_pred,
                            target_names=target_names))
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=CHESS_CHARS,
                      title='Confusion matrix, without normalization')
plt.show()
