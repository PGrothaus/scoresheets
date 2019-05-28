# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""
import sys
import io
import logging
import os
import json
import hashlib

import PIL.Image
import tensorflow as tf

import utilities.config as conf
from utilities.tf_flags import load_object_detection_parameters
from object_detection.utils import dataset_util
from datasets.character_recognition.base import (
    get_image_path,
    get_bounding_boxes,
    is_train_example,
)

fp = sys.argv[1]
conf.loadconfig(fp)

load_object_detection_parameters()
FLAGS = tf.app.flags.FLAGS


def create_tf_example(filename, boxes):
    # Populate the following variables from your example.
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    image = PIL.Image.open(encoded_jpg_io)
    height = image.height # Image height
    width = image.width # Image width

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(
            boxes.xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(
            boxes.xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(
            boxes.ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(
            boxes.ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            boxes.class_names),
        'image/object/class/label': dataset_util.int64_list_feature(
            boxes.class_labels),
        }))
    return tf_example


def main(_):
    path_train = os.path.join(FLAGS.output_dir, 'train.record')
    writer_train = tf.python_io.TFRecordWriter(path_train)

    path_eval = os.path.join(FLAGS.output_dir, 'eval.record')
    writer_eval = tf.python_io.TFRecordWriter(path_eval)

    # Write code to read in your dataset to examples variable
    data_dir = FLAGS.data_dir

    logging.info('Reading from scoresheet dataset.')
    image_dir = os.path.join(data_dir, 'images')
    annotations_path = os.path.join(data_dir, 'annotations.json')

    with open(annotations_path, "r") as f:
        annotations = json.loads(f)
    for annotation in annotations:
        image_path = get_image_path(annotation)
        boxes = get_bounding_boxes(annotation)
        example = create_tf_example(image_path, boxes)
        if is_train_example(annotation):
            tf_example = create_tf_example(image_dir, annotations_path, example)
            writer_train.write(tf_example.SerializeToString())
        else:
            tf_example = create_tf_example(image_dir, annotations_path, example)
            writer_eval.write(tf_example.SerializeToString())
    writer_train.close()
    writer_eval.close()


if __name__ == '__main__':
    tf.app.run()
