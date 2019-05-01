import unittest
import mock

from datasets.factory import build


class DatasetFactoryTestCase(unittest.TestCase):

    def test_factor_move_section_only(self):
        to_mock = "datasets.factory.build_move_section_only_dataset"
        with mock.patch(to_mock) as mocked_build:
            build("move-section-only")
            mocked_build.assert_called_once_with()

    def test_factor_unknown(self):
        dataset = build("thisKeyDoesNotExist")
        assert dataset is None
