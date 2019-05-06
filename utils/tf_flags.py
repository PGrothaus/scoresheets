import tensorflow as tf

import utils.config as conf

FLAGS = tf.app.flags.FLAGS


def load_east_flag_parameters():
    tf.app.flags.DEFINE_integer(
        'input_size', conf.get_param('input_size'), '')
    tf.app.flags.DEFINE_integer(
        'batch_size_per_gpu', conf.get_param('batch_size_per_gpu'), '')
    tf.app.flags.DEFINE_integer(
        'num_readers', conf.get_param('num_readers'), '')
    tf.app.flags.DEFINE_float(
        'learning_rate', conf.get_param('learning_rate'), '')
    tf.app.flags.DEFINE_integer(
        'max_steps', conf.get_param('max_steps'), '')
    tf.app.flags.DEFINE_float(
        'moving_average_decay', conf.get_param('moving_average_decay'), '')
    tf.app.flags.DEFINE_string(
        'gpu_list', conf.get_param('gpu_list'), '')
    tf.app.flags.DEFINE_string(
        'checkpoint_path', conf.get_param('checkpoint_path'), '')
    tf.app.flags.DEFINE_boolean(
        'restore', conf.get_param('restore'), 'whether to resotre from checkpoint')
    tf.app.flags.DEFINE_integer(
        'save_checkpoint_steps', conf.get_param('save_checkpoint_steps'), '')
    tf.app.flags.DEFINE_integer(
        'save_summary_steps', conf.get_param('save_summary_steps'), '')
    tf.app.flags.DEFINE_string(
        'pretrained_model_path', conf.get_param('pretrained_model_path'), '')
    tf.app.flags.DEFINE_string(
        'training_raw_data_path', conf.get_param('training_raw_data_path'), '')
    tf.app.flags.DEFINE_string(
        'training_batches_path', conf.get_param('training_batches_path'), '')
    tf.app.flags.DEFINE_string(
        'geometry', conf.get_param('geometry'), '')
    tf.app.flags.DEFINE_integer('max_image_large_side',
                                conf.get_param('max_image_large_side'),
                                'max image size of training')
    tf.app.flags.DEFINE_integer('max_text_size', conf.get_param('max_text_size'),
                                'if the text in the input image is bigger than this, then we resize'
                                'the image according to this')
    tf.app.flags.DEFINE_integer('min_text_size', conf.get_param('min_text_size'),
                                'if the text size is smaller than this, we ignore it during training')
    tf.app.flags.DEFINE_float('min_crop_side_ratio', conf.get_param('min_crop_side_ratio'),
                              'when doing random crop from input image, the'
                              'min length of min(H, W')
    tf.app.flags.DEFINE_integer('text_scale', conf.get_param('text_scale'), '')
    tf.app.flags.DEFINE_integer(
        'n_batches_to_create', conf.get_param('n_batches_to_create'), '')
    tf.app.flags.DEFINE_boolean(
        'visualise', conf.get_param("visualise"), '')
