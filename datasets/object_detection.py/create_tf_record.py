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
import random

import PIL.Image
import tensorflow as tf

import utilities.config as conf
from utilities.tf_flags import load_object_detection_parameters
from object_detection.utils import dataset_util

fp = sys.argv[1]
conf.loadconfig(fp)

load_object_detection_parameters()
FLAGS = tf.app.flags.FLAGS


def load_annotations(fp, width=None, height=None):
    with open(fp, 'r') as f:
        lines = f.readlines()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    labels = []
    class_names = []
    for line in lines:
        coordinates = line.split(',')[:8]
        x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        xs = list(map(int, xs))
        ys = list(map(int, ys))
        xmins.append(min(xs) / width)
        xmaxs.append(max(xs) / width)
        ymins.append(min(ys) / height)
        ymaxs.append(max(ys) / height)
        labels.append(1)
        class_names.append("move".encode('utf8'))
    return xmins, xmaxs, ymins, ymaxs, labels, class_names


def create_tf_example(image_dir, annotations_dir, example):
    # Populate the following variables from your example.
    img_path = os.path.join(image_dir, example+'.jpg')
    print(img_path)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    height = image.height # Image height
    width = image.width # Image width

    source_id = img_path.rsplit("/", 1)[1].split(".")[0]
    label_path = os.path.join(annotations_dir, example+'.txt')
    print(label_path)
    annotations = load_annotations(label_path, width=width, height=height)
    xmins, xmaxs, ymins, ymaxs, classes, classes_text = annotations
    print(len(xmins))

    filename = img_path # Filename of the image. Empty if image is not from file
    image_format = b'jpeg' # b'jpeg' or b'png'
    print(height, width, source_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(source_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
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
    annotations_dir = os.path.join(data_dir, 'annotations')
    examples_path = os.path.join(annotations_dir, 'trainval.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.8 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    for example in train_examples:
        tf_example = create_tf_example(image_dir, annotations_dir, example)
        writer_train.write(tf_example.SerializeToString())
    writer_train.close()

    for example in val_examples:
        tf_example = create_tf_example(image_dir, annotations_dir, example)
        writer_eval.write(tf_example.SerializeToString())
    writer_eval.close()


if __name__ == '__main__':
    tf.app.run()
