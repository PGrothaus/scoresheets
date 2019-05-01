from icdar import generator
import tensorflow as tf
import pickle
import os
import sys
import random
import numpy as np

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 2, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 10, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')


FLAGS = tf.app.flags.FLAGS

random.seed(42)
np.random.seed(42)


def save_batch(batch, count):
    outdir = FLAGS.training_data_path
    fp = os.path.join(outdir, '{}.pkl'.format(count))
    with open(fp, 'wb') as f:
        print('saved to {}'.format(fp))
        pickle.dump(batch, f)


def main(argv=None):
    n_batches = 750
    count = 0
    for batch in generator(batch_size=FLAGS.batch_size_per_gpu):
        print(count)
        save_batch(batch, 0+count)
        count += 1
        if count > n_batches:
            sys.exit()


if __name__ == '__main__':
     # use --training_data_path=/data/scoresheets/dataset/train/ --geometry=RBOX
    tf.app.run()
