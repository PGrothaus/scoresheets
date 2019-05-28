import unittest
from datasets.character_recognition.base import (
    get_image_path,
    get_bounding_boxes,
    is_train_example,
)


class GetImagePathTestCase(unittest.TestCase):

    def setUp(self):
        self.image_dir = '/base/'
        self.annotation = {
            "ID": "cjvv90v1whapn0866vf1j4sez",
            "DataRow ID": "cjvv77fsosd670ctn7en178p0",
            "Label": {"4": [{"geometry": [{"x": 140, "y": 12},
                                          {"x": 140, "y": 55},
                                          {"x": 177, "y": 55},
                                          {"x": 177, "y": 12}]}],
                      "a": [{"geometry": [{"x": 101, "y": 24},
                                          {"x": 101, "y": 54},
                                          {"x": 139, "y": 54},
                                          {"x": 139, "y": 24}]}]},
            "Project Name": "Moves",
            "External ID": "95.jpg",
            "Dataset Name": "Moves-Test"}

    def test_returns_correct_path_for_image_in_moves_test(self):
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/6f50d463-9496-4974-83db-2dc879b0c38f/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_0a0(self):
        self.annotation["Dataset Name"] = "Moves-0a0e39c7"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/0a0e39c7-19ea-425f-8ace-2035bbe9a325/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_00a(self):
        self.annotation["Dataset Name"] = "Moves-00a76a37"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/00a76a37-e5a6-4ac0-b4ee-8d987a778738/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_0ab(self):
        self.annotation["Dataset Name"] = "Moves-0ab70b7f"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/0ab70b7f-4d5e-41d5-8b29-2b2a834a94f3/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_0b3(self):
        self.annotation["Dataset Name"] = "Moves-0b3327d3"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/0b3327d3-55ec-444d-8b8d-091f92ccdc9c/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_0be(self):
        self.annotation["Dataset Name"] = "Moves-0be59ced"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/0be59ced-c3d2-47ca-afb7-ac3faa57ba63/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_8a1(self):
        self.annotation["Dataset Name"] = "Moves-8a129712"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/8a129712-b9fa-4e3f-91a5-d444747760ed/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_8b0(self):
        self.annotation["Dataset Name"] = "Moves-8b0b0cc2"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/8b080cc2-0663-4a26-8496-b67a3e3bb61d/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_set1(self):
        self.annotation["Dataset Name"] = "Moves-set1"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/set_1/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_set2(self):
        self.annotation["Dataset Name"] = "Moves-set2"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/set_2/95.jpg'
        assert expected == path

    def test_returns_correct_path_for_image_in_moves_set3(self):
        self.annotation["Dataset Name"] = "Moves-set3"
        path = get_image_path(self.annotation, self.image_dir)
        expected = '/base/set_3/95.jpg'
        assert expected == path


class GetBoundingBoxesTestCase(unittest.TesetCase):

    def setUp(self):
        self.annotation = {
            "ID": "cjvv90v1whapn0866vf1j4sez",
            "DataRow ID": "cjvv77fsosd670ctn7en178p0",
            "Label": {"4": [{"geometry": [{"x": 140, "y": 12},
                                          {"x": 140, "y": 55},
                                          {"x": 177, "y": 55},
                                          {"x": 177, "y": 12}]}],
                      "a": [{"geometry": [{"x": 101, "y": 24},
                                          {"x": 101, "y": 54},
                                          {"x": 139, "y": 54},
                                          {"x": 139, "y": 24}]}]},
            "Project Name": "Moves",
            "External ID": "95.jpg",
            "Dataset Name": "Moves-Test"}
        self.boxes = get_bounding_boxes(self.annotation)

    def test_correct_number_of_boxes(self):
        assert len(self.boxes) == 2

    def test_has_correct_xmins(self):
        self.boxes.xmins == [140, 101]

    def test_has_correct_xmaxs(self):
        self.boxes.xmaxs == [177, 139]

    def test_has_correct_ymins(self):
        self.boxes.ymins == [12, 24]

    def test_has_correct_ymaxs(self):
        self.boxes.ymaxs == [55, 54]

    def test_has_correct_class_names(self):
        self.boxes.class_names == ["4", "a"]

    def test_has_correct_class_labels(self):
        self.boxes.class_labels == [4, 9]


class IsTrainExampleTestCase(unittest.TestCase):

    def setUp(self):
        self.annotation = {
            "ID": "cjvv90v1whapn0866vf1j4sez",
            "DataRow ID": "cjvv77fsosd670ctn7en178p0",
            "Label": {"4": [{"geometry": [{"x": 140, "y": 12},
                                          {"x": 140, "y": 55},
                                          {"x": 177, "y": 55},
                                          {"x": 177, "y": 12}]}],
                      "a": [{"geometry": [{"x": 101, "y": 24},
                                          {"x": 101, "y": 54},
                                          {"x": 139, "y": 54},
                                          {"x": 139, "y": 24}]}]},
            "Project Name": "Moves",
            "External ID": "95.jpg",
            "Dataset Name": "Moves-Test"}

        def test_returns_correct_for_image_in_moves_test(self):
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_0a0(self):
            self.annotation["Dataset Name"] = "Moves-0a0e39c7"
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_00a(self):
            self.annotation["Dataset Name"] = "Moves-00a76a37"
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_0ab(self):
            self.annotation["Dataset Name"] = "Moves-0ab70b7f"
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_0b3(self):
            self.annotation["Dataset Name"] = "Moves-0b3327d3"
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_0be(self):
            self.annotation["Dataset Name"] = "Moves-0be59ced"
            assert is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_8a1(self):
            self.annotation["Dataset Name"] = "Moves-8a129712"
            assert not is_train_example(self.annotation)

        def test_returns_correct_for_image_in_moves_8b0(self):
            self.annotation["Dataset Name"] = "Moves-8b0b0cc2"
            assert not is_train_example(self.annotation)

        def test_returns_correct_for_test_image_in_other_moves(self):
            self.annotation["Dataset Name"] = "Moves-other"
            self.annotation["External ID"] = "9fbb017_4.jpg"
            assert not is_train_example(self.annotation)

        def test_returns_correct_for_train_image_in_other_moves(self):
            self.annotation["Dataset Name"] = "Moves-other"
            self.annotation["External ID"] = "7fbb017_4.jpg"
            assert is_train_example(self.annotation)
