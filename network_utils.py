import tensorflow as tf
import tensorflow.contrib.slim as slim

import os


def decode_depth(encoded_depth):
    encoded_depth = tf.cast(encoded_depth, tf.float32)
    r, g, b = tf.unstack(encoded_depth, axis=2)
    depth = r * 65536.0 + g * 256.0 + b  # decode depth image
    depth = tf.div(depth, tf.constant(10000.0))
    return depth


def encode_depth(depth):
    depth = tf.multiply(depth, tf.constant(10000.0))
    depth = tf.cast(depth, tf.uint16)
    r = depth / 255 / 255
    g = depth / 255
    b = depth % 255
    encoded_depth = tf.stack([r, g, b], axis=2)
    encoded_depth = tf.cast(encoded_depth, tf.uint8)
    return encoded_depth


def get_dataset(dataset_dir, num_readers, num_preprocessing_threads, hparams, reader=None):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/color': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/encoded_depth': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/label': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/label_aug': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    items_to_handlers = {
        'color': slim.tfexample_decoder.Image(image_key='image/color', shape=(320, 320, 3), channels=3),
        'encoded_depth': slim.tfexample_decoder.Image(image_key='image/encoded_depth', shape=(320, 320, 3), channels=3),
        'label': slim.tfexample_decoder.Image(image_key='image/label', shape=(40, 40, 3), channels=3),
        'label_aug': slim.tfexample_decoder.Image(image_key='image/label_aug', shape=(40, 40, 3), channels=3),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=3,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              common_queue_capacity=20 * hparams.batch_size,
                                                              common_queue_min=10 * hparams.batch_size)
    color, encoded_depth, label, label_aug = provider.get(['color', 'encoded_depth', 'label', 'label_aug'])
    color = tf.cast(color, tf.float32)
    color = color / tf.constant([255.0])
    color = (color - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    depth = decode_depth(encoded_depth)
    depth = tf.stack([depth, depth, depth], axis=2)
    label = tf.cast(label, tf.float32)
    label_aug = tf.cast(label_aug, tf.float32)

    colors, depths, labels, label_augs = tf.train.batch([color, depth, label, label_aug],
                                                        batch_size=hparams.batch_size,
                                                        num_threads=num_preprocessing_threads,
                                                        capacity=5*hparams.batch_size)
    return colors, depths, labels, label_augs


def create_loss(net, labels, lamb):
    bad, good, background = tf.unstack(labels, axis=3)
    mask = lamb * tf.add(bad, good) + background
    attention_mask = tf.stack([mask, mask, mask], axis=3)
    y = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    cross_entropy = -tf.reduce_mean(attention_mask * (labels * tf.log(y)))
    return cross_entropy


def add_summary(colors, depths, label_augs, end_points, loss, scope='arcnet'):
    tf.summary.scalar('loss', loss)
    tf.summary.image('colors', colors)
    tf.summary.image('depths', depths)
    tf.summary.image('label_augs', label_augs)
    # for i in range(1, 3):
    #     for j in range(64):
    #         tf.summary.image(scope + '/conv{}' + '_{}'.format(i, j),
    #                          end_points[scope + '/conv{}'.format(i)][0:1, :, :, j:j + 1])
    tf.summary.image(scope + '/conv3', end_points[scope + '/conv3'])
    net = end_points['logits']
    tf.summary.image('inference_map', tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True))
    variable_list = slim.get_model_variables()
    for var in variable_list:
        tf.summary.histogram(var.name[:-2], var)


def restore_map():
    variable_list = slim.get_model_variables()
    variables_to_restore = {var.op.name: var for var in variable_list}
    return variables_to_restore


def restore_from_classification_checkpoint(color_scope, depth_scope, model_name, checkpoint_exclude_scope):
    color_variable_list = slim.get_model_variables(color_scope)
    color_variable_list = [var for var in color_variable_list if checkpoint_exclude_scope not in var.op.name]
    depth_variable_list = slim.get_model_variables(depth_scope)
    depth_variable_list = [var for var in depth_variable_list if checkpoint_exclude_scope not in var.op.name]
    color_variables_to_restore = {}
    depth_variables_to_restore = {}
    for var in color_variable_list:
        if var.name.startswith(color_scope):
            var_name = var.op.name.replace(color_scope, model_name)
            color_variables_to_restore[var_name] = var
    for var in depth_variable_list:
        if var.name.startswith(depth_scope):
            var_name = var.op.name.replace(depth_scope, model_name)
            depth_variables_to_restore[var_name] = var
    return color_variables_to_restore, depth_variables_to_restore
