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

    tf.app.flags.DEFINE_string(
        'test_data_path', conf.get_param("test_data_path"), '')
    tf.app.flags.DEFINE_string(
        'output_dir', conf.get_param("output_dir"), '')
    tf.app.flags.DEFINE_bool(
        'no_write_images', conf.get_param("no_write_images"), 'dont write images')
    tf.app.flags.DEFINE_integer(
        'polygons_per_example', conf.get_param("polygons_per_example"), None)


def load_object_detection_parameters():
    tf.app.flags.DEFINE_string('data_dir',
        conf.get_param("data_dir"), 'Root directory to raw pet dataset.')
    tf.app.flags.DEFINE_string('output_dir',
        conf.get_param("output_dir"), 'Path to directory to output TFRecords.')
    tf.app.flags.DEFINE_string('label_map_path',
        conf.get_param("label_map_path"), 'Path to label map proto')
    tf.app.flags.DEFINE_boolean('faces_only',
        conf.get_param("faces_only"), 'If True, generates bounding boxes '
                         'for pet faces.  Otherwise generates bounding boxes (as '
                         'well as segmentations for full pet bodies).  Note that '
                         'in the latter case, the resulting files are much larger.')
    tf.app.flags.DEFINE_string('mask_type',
        conf.get_param("mask_type"), 'How to represent instance '
                        'segmentation masks. Options are "png" or "numerical".')
    tf.app.flags.DEFINE_integer('num_shards',
        conf.get_param("num_shards"), 'Number of TFRecord shards')


    tf.app.flags.DEFINE_string(
        'model_dir', conf.get_param("model_dir"), 'Path to output model directory '
        'where event and checkpoint files will be written.')
    tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                        'file.')
    tf.app.flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
    tf.app.flags.DEFINE_boolean('eval_training_data', False,
                         'If training data should be evaluated for this job. Note '
                         'that one call only use this in eval-only mode, and '
                         '`checkpoint_dir` must be supplied.')
    tf.app.flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                         'every n eval input examples, where n is provided.')
    tf.app.flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                         'one of every n train input examples for evaluation, '
                         'where n is provided. This is only used if '
                         '`eval_training_data` is True.')
    tf.app.flags.DEFINE_string(
        'hparams_overrides', None, 'Hyperparameter overrides, '
        'represented as a string containing comma-separated '
        'hparam_name=value pairs.')
    tf.app.flags.DEFINE_string(
        'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
        '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
        'writing resulting metrics to `model_dir`.')
    tf.app.flags.DEFINE_boolean(
        'run_once', False, 'If running in eval-only mode, whether to run just '
        'one round of eval vs running continuously (default).'
    )


def load_character_recognition_parameters():
    tf.app.flags.DEFINE_string('data_dir',
        conf.get_param("data_dir"), 'Root directory to raw pet dataset.')
    tf.app.flags.DEFINE_string('output_dir',
        conf.get_param("output_dir"), 'Path to directory to output TFRecords.')
    tf.app.flags.DEFINE_string('label_map_path',
        conf.get_param("label_map_path"), 'Path to label map proto')
    tf.app.flags.DEFINE_boolean('faces_only',
        conf.get_param("faces_only"), 'If True, generates bounding boxes '
                         'for pet faces.  Otherwise generates bounding boxes (as '
                         'well as segmentations for full pet bodies).  Note that '
                         'in the latter case, the resulting files are much larger.')
    tf.app.flags.DEFINE_string('mask_type',
        conf.get_param("mask_type"), 'How to represent instance '
                        'segmentation masks. Options are "png" or "numerical".')
    tf.app.flags.DEFINE_integer('num_shards',
        conf.get_param("num_shards"), 'Number of TFRecord shards')


    tf.app.flags.DEFINE_string(
        'model_dir', conf.get_param("model_dir"), 'Path to output model directory '
        'where event and checkpoint files will be written.')
    tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                        'file.')
    tf.app.flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
    tf.app.flags.DEFINE_boolean('eval_training_data', False,
                         'If training data should be evaluated for this job. Note '
                         'that one call only use this in eval-only mode, and '
                         '`checkpoint_dir` must be supplied.')
    tf.app.flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                         'every n eval input examples, where n is provided.')
    tf.app.flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                         'one of every n train input examples for evaluation, '
                         'where n is provided. This is only used if '
                         '`eval_training_data` is True.')
    tf.app.flags.DEFINE_string(
        'hparams_overrides', None, 'Hyperparameter overrides, '
        'represented as a string containing comma-separated '
        'hparam_name=value pairs.')
    tf.app.flags.DEFINE_string(
        'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
        '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
        'writing resulting metrics to `model_dir`.')
    tf.app.flags.DEFINE_boolean(
        'run_once', False, 'If running in eval-only mode, whether to run just '
        'one round of eval vs running continuously (default).'
    )
