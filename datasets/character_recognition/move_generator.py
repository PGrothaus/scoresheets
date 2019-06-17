import os
import pickle
import sys
import random
from datasets.character_recognition import label_map
import datasets.emnist as emnist
from datasets.emnist import Example
from collections import namedtuple

GeneratedExample = namedtuple("GeneratedExample", ["image", "position"])


def load_moves(data_dir):
    moves_path = os.path.join(data_dir, 'moves.pkl')
    with open(moves_path, "rb") as f:
        return pickle.load(f)


def get_chars_group(char):
    groups = {
        "1": "number",
        "2": "number",
        "3": "number",
        "4": "number",
        "5": "number",
        "6": "number",
        "7": "number",
        "8": "number",
        "a": "lowercase",
        "b": "lowercase",
        "c": "lowercase",
        "d": "lowercase",
        "e": "lowercase",
        "f": "lowercase",
        "g": "lowercase",
        "h": "lowercase",
        "R (rook)": "uppercase",
        "N (knight)": "uppercase",
        "B (bishop)": "uppercase",
        "Q (queen)": "uppercase",
        "K (king)": "uppercase",
        "x (takes)": "x",
        "0 (rochade)": "0",
        "- (rochade)": "-",
        "+ (check)": "+",
        "= (promoting or draw offer)": "=",
        }
    return groups[char]


def randomise_char(char):
    char = get_chars_group(char)
    if char == "number":
        return random.choice(["1", "2", "3", "4", "5", "6", "7", "8"])
    elif char == "lowercase":
        return random.choice(["a", "b", "c", "d", "e", "f", "g", "h"])
    elif char == "uppercase":
        return random.choice(["R", "N", "B", "Q", "K"])
    elif char == "x":
        return "x"
    elif char == "0":
        return "0"
    else:
        return None


def randomise_move(move):
    print(move)
    if move:
        move = [randomise_char(char) for char in list(move)]
        return [c for c in move if c]


def represent_move(move, examples):
    move = randomise_move(move)
    if move:
        move = [get_random_char_from_dataset(char, examples) for char in move]
        positions = [None] * len(move)
        positions[0] = "is-first"
        positions[-1] = "is-last"
        group = zip(move, positions)
        return [GeneratedExample(img, pos) for img, pos in group]


def get_random_char_from_dataset(char, examples):
    print(char)
    return random.choice(examples[char])


def generate_moves(data_dir, key, n_moves=10):
    images = []
    moves = load_moves(data_dir)
    examples = emnist.load(key)
    print(len(examples["a"]))
    COND = True
    while COND:
        for move in moves:
            data = represent_move(move, examples)
            if data:
                images.extend(data)
            if len(images) >= n_moves:
                COND = False
                break
    print(len(images))
    with open(data_dir+"generate_moves.{}.pkl".format(key), "wb") as f:
        pickle.dump(images, f)


def load_generated_moves(key):
    data_dir = "/data/scoresheets/dataset/character-detection/"
    with open(data_dir+"generate_moves.{}.pkl".format(key), "rb") as f:
        return pickle.load(f)
    
if "__main__" == __name__:
    generate_moves(sys.argv[1], 'train', n_moves=50000)
    generate_moves(sys.argv[1], 'test', n_moves=20000)
