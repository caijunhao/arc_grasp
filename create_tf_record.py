import tensorflow as tf
import random

import argparse
import os

import dataset_utils

parser = argparse.ArgumentParser(description='create tensorflow record')
parser.add_argument('--set', default='train-processed-split', type=str, help='Convert train or test set.')
parser.add_argument('--data_path', default='training', type=str, help='Path to data set.')
args = parser.parse_args()

folders = ['color', 'encoded_depth', 'label_map', 'label_aug_map']


def dict_to_tf_example(file_name):
    with open(os.path.join(args.data_path, folders[0], file_name+'.png'), 'rb') as fid:
        encoded_color = fid.read()
    with open(os.path.join(args.data_path, folders[1], file_name + '.png'), 'rb') as fid:
        encoded_depth = fid.read()
    with open(os.path.join(args.data_path, folders[2], file_name + '.png'), 'rb') as fid:
        encoded_label_map = fid.read()
    with open(os.path.join(args.data_path, folders[3], file_name + '.png'), 'rb') as fid:
        encoded_label_aug_map = fid.read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/color': dataset_utils.bytes_feature(encoded_color),
        'image/format': dataset_utils.bytes_feature('png'),
        'image/encoded_depth': dataset_utils.bytes_feature(encoded_depth),
        'image/label': dataset_utils.bytes_feature(encoded_label_map),
        'image/label_aug': dataset_utils.bytes_feature(encoded_label_aug_map),
    }))
    return example


def main():
    with open(os.path.join(args.data_path, args.set+'.txt'), 'r') as f:
        file_list = f.readlines()
    random.shuffle(file_list)
    writer = tf.python_io.TFRecordWriter(os.path.join(args.data_path, 'train.tfrecord'))
    for file_name in file_list:
        tf_example = dict_to_tf_example(file_name[:-1])
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()

