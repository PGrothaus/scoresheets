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
from utils import plot_confusion_matrix
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels


label_to_id_map = {lbl: i for i, lbl in enumerate(CHESS_CHARS)}
num_classes = len(label_to_id_map)


def prep_data(key):
    examples = load_generated_moves(key)

    #train_data = load(key)
    #examples = []
    #for k, v in train_data.items():
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
#model.add(keras.layers.Conv2D(
#    filters=128, kernel_size=(2, 2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['categorical_accuracy'])
model.summary()


data_generator_train = keras.preprocessing.image.ImageDataGenerator()
data_generator_valid = keras.preprocessing.image.ImageDataGenerator()

#imgs_test = imgs_train
#targets_test = targets_train
#    validation_split=.2)
## consider using this for more variety
#data_generator = keras.preprocessing.image.ImageDataGenerator(
#    validation_split=.2, width_shift_range=.2,
#    height_shift_range=.2, rotation_range=60,
#    zoom_range=.2, shear_range=.3)


training_data_generator = data_generator_train.flow(imgs_train, targets_train) #, subset='training')
validation_data_generator = data_generator_valid.flow(imgs_test, targets_test, shuffle=False) #, subset='validation')
history = model.fit_generator(training_data_generator,
                              steps_per_epoch=500, epochs=5,
                              validation_data=validation_data_generator)

Y_pred = model.predict_generator(validation_data_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(targets_test, axis=1)
print(y_test.shape, y_pred.shape, Y_pred.shape, targets_test.shape)
print(y_pred.shape)
print(targets_test.shape)
print('Confusion Matrix')
print(confusion_matrix(np.argmax(targets_test, axis=1), y_pred))
print('Classification Report')
target_names = CHESS_CHARS
print(classification_report(np.argmax(targets_test, axis=1), y_pred, target_names=target_names))


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=CHESS_CHARS,
                      title='Confusion matrix, without normalization')

plt.show()
