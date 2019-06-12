import random
from datasets.emnist import (
    Example,
    load,
    plot_example,
    CHESS_CHARS,
)


data = load('test')
for char in CHESS_CHARS:
    examples = data[char]
    example = random.choice(examples)
    plot_example(example)
