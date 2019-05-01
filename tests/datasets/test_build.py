import unittest

from datasets.build import (
    shift_polygon,
    in_move_section,
    order_box,
    filter_by_annotator,
    )


class ShiftPolygonTestCase(unittest.TestCase):

    def setUp(self):
        self.polygon = [0, 0, 5, 0, 5, 5, 0, 5]
        self.move_section = [-1, -1]

    def test_shift_polygon(self):
        shifted_polygon = shift_polygon(self.polygon, self.move_section)
        expected_polygon = [1, 1, 6, 1, 6, 6, 1, 6]
        assert expected_polygon == shifted_polygon


class InMoveSectionTestCase(unittest.TestCase):

    def setUp(self):
        self.move_section = [0, 0, 5, 0, 5, 5, 0, 5]

    def test_above_move_section(self):
        polygon = [0, -3, 5, -3, 5, -1, 0, -1]
        assert not in_move_section(self.move_section, polygon)

    def test_below_move_section(self):
        polygon = [0, 7, 5, 7, 5, 10, 0, 10]
        assert not in_move_section(self.move_section, polygon)

    def test_left_of_move_section(self):
        polygon = [-5, 0, -1, 0, -1, 5, -5, 5]
        assert not in_move_section(self.move_section, polygon)

    def test_right_of_move_section(self):
        polygon = [6, 0, 8, 0, 8, 5, 6, 5]
        assert not in_move_section(self.move_section, polygon)

    def test_in_move_section(self):
        polygon = [1, 1, 4, 1, 4, 4, 1, 4]
        assert in_move_section(self.move_section, polygon)

    def test_bordering_move_section(self):
        polygon = [0, 0, 3, 0, 3, 2, 0, 2]
        assert in_move_section(self.move_section, polygon)


class OrderBoxTestCase(unittest.TestCase):

    def test_correct_order(self):
        bbox = {"geometry": [
            {'x': 0, 'y': 0},
            {'x': 5, 'y': 0},
            {'x': 5, 'y': 5},
            {'x': 0, 'y': 5},
            ]}
        result = order_box(bbox)
        expected = [0, 0, 5, 0, 5, 5, 0, 5]
        assert expected == result

    def test_wrong_order(self):
        bbox = {"geometry": [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 0},
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 5},
            ]}
        result = order_box(bbox)
        expected = [0, 0, 5, 0, 5, 5, 0, 5]
        assert expected == result


class FilterByAnnotatorTestCase(unittest.TestCase):

    def setUp(self):
        self.raw_data = [
            {"Created_By": "collaborator"},
            {"Created_By": "other"},
            {"Created_By": "collaborator"},
            ]

    def self_test_filter_by_annotator(self):
        result = filter_by_annotator(raw_data)
        expected = [{"Created_By": "collaborator"},
                    {"Created_By": "collaborator"}]
        assert result == expected
