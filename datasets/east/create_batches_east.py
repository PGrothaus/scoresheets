import sys
from external.east.icdar import generator
import tensorflow as tf
import pickle
import os
import random
import numpy as np

from progressbar import ProgressBar

import utils.config as conf
from utils.tf_flags import load_east_flag_parameters

fp_config = sys.argv[1]
conf.loadconfig(fp_config)

load_east_flag_parameters()
FLAGS = tf.app.flags.FLAGS

random.seed(conf.get_param('seed'))
np.random.seed(conf.get_param('seed'))


def save_batch(batch, count):
    outdir = FLAGS.training_batches_path
    fp = os.path.join(outdir, '{}.pkl'.format(count))
    with open(fp, 'wb') as f:
        pickle.dump(batch, f)


def main(argv=None):
    n_batches = FLAGS.n_batches_to_create
    count = 537
    with ProgressBar(max_value=n_batches) as bar:
        while count < n_batches:
            for batch in generator(batch_size=FLAGS.batch_size_per_gpu,
                                   vis=FLAGS.visualise):
                save_batch(batch, 0+count)
                count += 1
                bar.update(count)


if __name__ == '__main__':
    tf.app.run()
