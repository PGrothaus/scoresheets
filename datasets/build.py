import os
import json

from PIL import Image


path = '/data/scoresheets/annotations/annotated_sheets_25022019.json'


def build_move_section_only_dataset():
    # TODO: Implement this (use code from notebook)
    global path
    raw_data = get_raw_data(path)
    data = filter_by_annotator(raw_data)
    for i, item in enumerate(data):
        print(i, item["External ID"])
        create_target_file_east_detector_move_section_only(item)


def get_raw_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def filter_by_annotator(raw_data):
    filtered_data = []
    for row in raw_data:
        annotator = row["Created By"]
        if annotator.startswith("collaborator"):
            filtered_data.append(row)
    return filtered_data


def create_target_file_east_detector_move_section_only(item, outdir='/data/scoresheets/dataset/move-section-only/'):
    try:
        move_section = item["Label"]["Move-Section"][0]
    except:
        print("no move section!!!")
        return None
    move_section_area = order_box(move_section)
    labels = item["Label"]
    item_id = item["External ID"].split(".jpg")[0]
    fp = '/data/scoresheets/dataset/all_annotated_images/' + str(item_id) + '.jpg'
    polygons = []
    for box_type, bboxes in labels.items():
        if box_type in ["Move-Section", "Header"]:
            continue
        for bbox in bboxes:
            pg = order_box(bbox)
            if in_move_section(move_section_area, pg):
                pg = shift_polygon(pg, move_section_area)
                polygons.append((box_type, pg))
    outfile = os.path.join(outdir, "{}.txt".format(item_id))
    with open(outfile, "w") as out:
        for t1 in polygons:
            box_type, polygon = t1
            polygon = map(str, polygon)
            line = ",".join(polygon) + ",{}\n".format(box_type.replace(" ", ""))
            out.write(line)
    crop_image(fp, move_section_area)


def order_box(bbox):
    x1, y1 = bbox["geometry"][0]['x'], bbox["geometry"][0]['y']
    x2, y2 = bbox["geometry"][1]['x'], bbox["geometry"][1]['y']
    x3, y3 = bbox["geometry"][2]['x'], bbox["geometry"][2]['y']
    x4, y4 = bbox["geometry"][3]['x'], bbox["geometry"][3]['y']
    mi_x = min([x1, x2, x3, x4])
    ma_x = max([x1, x2, x3, x4])
    mi_y = min([y1, y2, y3, y4])
    ma_y = max([y1, y2, y3, y4])
    return [mi_x, mi_y, ma_x, mi_y, ma_x, ma_y, mi_x, ma_y]


def in_move_section(move_section, polygon):
    in_area = False
    if polygon[0] >= move_section[0]:
        if polygon[1] >= move_section[1]:
            if polygon[2] <= move_section[2]:
                if polygon[-1] <= move_section[-1]:
                    in_area = True
    return in_area


def shift_polygon(pg, move_section):
    dx = move_section[0]
    dy = move_section[1]
    return [pg[0] - dx,
            pg[1] - dy,
            pg[2] - dx,
            pg[3] - dy,
            pg[4] - dx,
            pg[5] - dy,
            pg[6] - dx,
            pg[7] - dy,
           ]


def crop_image(fp, move_section):
    img = Image.open(fp)
    img = img.crop((move_section[0], move_section[1], move_section[2], move_section[-1]))
    outdir = '/data/scoresheets/dataset/move-section-only/'
    fp_out = outdir + fp.rsplit('/', 1)[-1]
    img.save(fp_out)
