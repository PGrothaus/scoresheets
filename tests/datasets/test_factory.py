import unittest
from datasets.factory import build


class DatasetFactoryTestCase(unittest.TestCase):

    def test_factor_move_section_only(self):
        dataset = build("move-section-only")
        assert 5 == dataset

    def test_factor_unknown(self):
        dataset = build("thisKeyDoesNotExist")
        assert dataset is None
