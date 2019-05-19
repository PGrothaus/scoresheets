import os
from PIL import Image

from datasets.build import (
    get_raw_data,
    filter_by_annotator,
    order_box,
)


path = '/data/scoresheets/annotations/annotated_sheets_25022019.json'


def extract_all_individual_moves():
    global path
    raw_data = get_raw_data(path)
    data = filter_by_annotator(raw_data)
    for i, datum in enumerate(data):
        print(i, datum["External ID"])
        for i, item in enumerate(extract_moves(datum)):
            fp, polygon = item
            crop_image(fp, polygon, i)


def extract_moves(item):
    labels = item["Label"]
    for box_type, bboxes in labels.items():
        item_id = item["External ID"].split(".jpg")[0]
        fp = '/data/scoresheets/dataset/all_annotated_images/' + str(item_id) + '.jpg'
        if box_type == "Move":
            for bbox in bboxes:
                yield fp, order_box(bbox)


def crop_image(fp, move_section, i):
    img = Image.open(fp)
    img = img.crop((move_section[0], move_section[1], move_section[2], move_section[-1]))
    base = fp.rsplit('/', 1)[-1].split(".jpg")[0]
    outdir = '/data/scoresheets/dataset/individual_moves/' + base +'/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fp_out = outdir + str(i) + ".jpg"
    img.save(fp_out)


if "__main__" == __name__:
    extract_all_individual_moves()
