import csv
import pickle
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

path_raw_train = '/data/scoresheets/dataset/EMNIST/emnist-byclass-train.csv'
path_raw_test = '/data/scoresheets/dataset/EMNIST/emnist-byclass-test.csv'

path_train = '/data/scoresheets/dataset/EMNIST/chars_train.pkl'
path_test = '/data/scoresheets/dataset/EMNIST/chars_test.pkl'

Example = namedtuple("Example", ["image", "label"])


CHESS_CHARS = ["K", "Q", "R", "B", "N",
               "a", "b", "c", "d", "e", "f", "g", "h",
               "1", "2", "3", "4", "5", "6", "7", "8",
               "x", "0"]


map_string = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
id_to_label_map = {i: char for i, char in enumerate(map_string)}


def build():
    collect_and_save_relevant_examples('train')
    collect_and_save_relevant_examples('test')


def load(key):
    outpath = path_train if key == 'train' else path_test
    with open(outpath, "rb") as f:
        return pickle.load(f)


def collect_and_save_relevant_examples(key, n_max=10000):
    examples = load_images_from_emnist(CHESS_CHARS, key, n_max=n_max)
    outpath = path_train if key == 'train' else path_test
    with open(outpath, "wb") as f:
        pickle.dump(examples, f)


def load_images_from_emnist(list_of_chars, key='train', n_max=100):
    examples = {}
    counts = {c: 0 for c in list_of_chars}
    for img, target in generate_emnist_images(key):
        label = id_to_label_map.get(target)
        if label in list_of_chars:
            if counts[label] < n_max:
                examples.setdefault(label, []).append(Example(img, label))
                counts[label] += 1
        if all([counts[c] == n_max for c in list_of_chars]):
            break
    return examples


def generate_emnist_images(key):
    assert key in ['train', 'test']
    path = path_raw_train if key == 'train' else path_raw_test
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            target = int(row[0])
            img = list(map(int, row[1:]))
            img = np.asarray(img).reshape((28, 28))
            img = np.transpose(img, axes=[1, 0])
            yield img[None], target


def plot_example(example):
    plt.title('Class: ' + str(example.label))
    data = example.image[0]
    plt.imshow(data, cmap='Greys_r')
    plt.show()


if "__main__" == __name__:
    collect_and_save_relevant_examples('train')
    collect_and_save_relevant_examples('test')
