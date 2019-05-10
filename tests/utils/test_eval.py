import unittest
import pytest
import shapely

from utils.eval_east import (
    create_polygon,
    get_iou_for_two_polygons,
    read_polygon_file,
    process_model_polygon,
    )


class InitPolygontestCase(unittest.TestCase):

    def setUp(self):
        self.coordinates = [1, 1, 3, 1, 3, 3, 1, 3]

    def test_init_polygon(self):
        polygon = create_polygon(*self.coordinates)
        assert isinstance(polygon, shapely.geometry.Polygon)


class PolygontestCase(unittest.TestCase):

    def setUp(self):
        self.coordinates = [1, 1, 3, 1, 3, 3, 1, 3]
        self.polygon = create_polygon(*self.coordinates)

    def test_polygon_area(self):
        assert self.polygon.area == 4.


class IuOTestCase(unittest.TestCase):

    def setUp(self):
        self.coordinates1 = [1, 1, 3, 1, 3, 3, 1, 3]
        self.polygon1 = create_polygon(*self.coordinates1)

        self.coordinates2 = [2, 1, 4, 1, 4, 3, 2, 3]
        self.polygon2 = create_polygon(*self.coordinates2)

    def test_iuo_value(self):
        result = get_iou_for_two_polygons(self.polygon1, self.polygon2)
        expected = 2. / 6.
        assert result == expected


class ReadPolygonFileTestCase(unittest.TestCase):

    def test_read_file(self):
        fp = './tests/test_files/coordinates.txt'
        polygons = read_polygon_file(fp)
        assert len(polygons) == 3
        iou = get_iou_for_two_polygons(polygons[0], polygons[2])
        assert iou == 2. / 6.


class ProcessPolygonTestCase(unittest.TestCase):

    def setUp(self):
        fp = './tests/test_files/coordinates.txt'
        self.polygons = read_polygon_file(fp)

    def test_matched_polygon(self):
        result = process_model_polygon(self.polygons[0], self.polygons)
        expected = 1, 0
        assert expected == result

    def test_unmatched_polygon(self):
        p0 = create_polygon(100, 100, 101, 100, 101, 102, 100, 102)
        expected = 0, None
        result = process_model_polygon(p0, self.polygons)
        assert expected == result
