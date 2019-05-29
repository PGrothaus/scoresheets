import os
from collections import namedtuple


Box = namedtuple("Box", ["xmins",
                         "xmaxs",
                         "ymins",
                         "ymaxs",
                         "class_names",
                         "class_labels"])


label_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "a": 9,
    "b": 10,
    "c": 11,
    "d": 12,
    "e": 13,
    "f": 14,
    "g": 15,
    "h": 16,
    "R (rook)": 17,
    "N (knight)": 18,
    "B (bishop)": 19,
    "Q (queen)": 20,
    "K (king)": 21,
    "x (takes)": 22,
    "0 (rochade)": 23,
    "- (rochade)": 24,
    "+ (check)": 25,
    "= (promoting or draw offer)": 26
}


def get_image_path(annotation, image_dir):
    directories = {
        "Moves-Test": "6f50d463-9496-4974-83db-2dc879b0c38f",
        "Moves-0a0e39c7": "0a0e39c7-19ea-425f-8ace-2035bbe9a325",
        "Moves-00a76a37": "00a76a37-e5a6-4ac0-b4ee-8d987a778738",
        "Moves-0ab70b7f": "0ab70b7f-4d5e-41d5-8b29-2b2a834a94f3",
        "Moves-0b3327d3": "0b3327d3-55ec-444d-8b8d-091f92ccdc9c",
        "Moves-0be59ced": "0be59ced-c3d2-47ca-afb7-ac3faa57ba63",
        "Moves-8a129712": "8a129712-b9fa-4e3f-91a5-d444747760ed",
        "Moves-8b0b0cc2": "8b080cc2-0663-4a26-8496-b67a3e3bb61d",
        "Moves-set1": "set_1",
        "Moves-set2": "set_2",
        "Moves-set3": "set_3",
    }
    dataset_key = annotation["Dataset Name"]
    dataset_dir = directories[dataset_key]
    image_id = annotation["External ID"]
    return os.path.join(image_dir, dataset_dir, image_id)


def get_bounding_boxes(annotation):
    global label_map
    drawn_boxes = annotation["Label"]
    if drawn_boxes == "Skip":
        return None
    bounding_box = Box([], [], [], [], [], [])
    for class_name, boxes in drawn_boxes.items():
        class_name = "K (king)" if class_name.startswith("K") else class_name
        class_name = "+ (check)" if class_name.startswith("+ (") else class_name
        class_label = label_map[class_name]
        for box in boxes:
            coordinates = box["geometry"]
            xs = [c["x"] for c in coordinates]
            ys = [c["y"] for c in coordinates]
            bounding_box.xmins.append(min(xs))
            bounding_box.xmaxs.append(max(xs))
            bounding_box.ymins.append(min(ys))
            bounding_box.ymaxs.append(max(ys))
            bounding_box.class_names.append(class_name)
            bounding_box.class_labels.append(class_label)
    return bounding_box


def is_train_example(annotation):
    cached = {
        "Moves-Test": True,
        "Moves-0a0e39c7": True,
        "Moves-00a76a37": True,
        "Moves-0ab70b7f": True,
        "Moves-0b3327d3": True,
        "Moves-0be59ced": True,
        "Moves-8a129712": False,
        "Moves-8b0b0cc2": False,
    }
    dataset_name = annotation["Dataset Name"]
    is_train = cached.get(dataset_name)
    if is_train is None:
        external_id = annotation["External ID"]
        if external_id.startswith("8"):
            is_train = False
        elif external_id.startswith("9"):
            is_train = False
        else:
            is_train = True
    return is_train
