from hparams import create_hparams
from network_utils import get_dataset
from model import model

import tensorflow as tf
import numpy as np

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='network evaluating')
parser.add_argument('--dataset_dir', default='', type=str, help='The directory where the datasets can be found.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='The directory where the checkpoint can be found')
args = parser.parse_args()


def main():
    hparams = create_hparams()
    colors, depths, labels, label_augs = get_dataset(args.dataset_dir,
                                                     num_readers=1,
                                                     num_preprocessing_threads=1,
                                                     hparams=hparams,
                                                     shuffle=False,
                                                     num_epochs=1,
                                                     is_training=False)
    net, end_points = model(colors,
                            depths,
                            num_classes=3,
                            num_channels=1000,
                            is_training=False,
                            global_pool=False,
                            output_stride=16,
                            spatial_squeeze=False,
                            color_scope='color_tower',
                            depth_scope='depth_tower',
                            scope='arcnet')
    probability_map = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    print 'Successfully loading model: {}.'.format(tf.train.latest_checkpoint(args.checkpoint_dir))
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    try:
        while not coord.should_stop():
            sample_result, sample_label = sess.run([probability_map, label_augs])
            if np.sum((sample_label[..., 0:2])) == 0:
                continue
            threshold = np.max(sample_result[..., 1]) - 0.0001
            sample_tp = (sample_result[..., 1] > threshold) & (sample_label[..., 1] == 1)
            sample_fp = (sample_result[..., 1] > threshold) & (sample_label[..., 0] == 1)
            sample_tn = (sample_result[..., 1] <= threshold) & (sample_label[..., 0] == 1)
            sample_fn = (sample_result[..., 1] <= threshold) & (sample_label[..., 1] == 1)
            tp += np.sum(sample_tp)
            fp += np.sum(sample_fp)
            tn += np.sum(sample_tn)
            fn += np.sum(sample_fn)
    except tf.errors.OutOfRangeError:
        print 'epoch limit reached.'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print 'precision : %f' % precision
    print 'recall : %f' % recall


if __name__ == '__main__':
    main()
