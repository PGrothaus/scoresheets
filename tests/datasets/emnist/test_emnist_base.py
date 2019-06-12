import unittest

import numpy as np

from datasets.emnist.base import CHESS_CHARS, id_to_label_map
from datasets.emnist.base import (
    load_images_from_emnist,
    Example,
)


def test_list_of_chess_chars():
    expected = ["K", "Q", "R", "B", "N",
                "a", "b", "c", "d", "e", "f", "g", "h",
                "1", "2", "3", "4", "5", "6", "7", "8",
                "x", "0"]
    assert sorted(CHESS_CHARS) == sorted(expected)


class IDToLabelMapTestCase(unittest.TestCase):

    def setUp(self):
        self.map = {v: k for k, v in id_to_label_map.items()}

    def test_K(self):
        assert self.map["K"] == 20


class LoadImagesFromEMNISTTestCase(unittest.TestCase):

    def setUp(self):
        self.n_max = 4
        self.chars = ["B", "N"]
        self.result = load_images_from_emnist(self.chars, 'test', self.n_max)

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_has_chars_as_keys(self):
        assert sorted(list(self.result.keys())) == self.chars

    def test_has_list_as_values(self):
        assert isinstance(self.result["B"], list)

    def test_has_max_items_in_list(self):
        assert len(self.result["B"]) == self.n_max

    def test_has_examples_in_list(self):
        assert isinstance(self.result["B"][0], Example)

    def test_array_has_correct_dimensions(self):
        assert self.result["B"][0].image.shape == (1, 28, 28)

    def test_other_array_has_correct_dimensions(self):
        assert self.result["N"][0].image.shape == (1, 28, 28)

    def test_max_pixel_value_in_correct_range(self):
        assert np.max(self.result["B"][0].image) <= 255

    def test_min_pixel_value_in_correct_range(self):
        assert np.min(self.result["B"][0].image) >= 0
