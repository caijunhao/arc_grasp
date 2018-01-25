from model import model

import tensorflow as tf
import numpy as np
import cv2

import scipy.io as sio
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='network evaluate.')
parser.add_argument('--height_map_color', required=True, type=str, help='color height map image.')
parser.add_argument('--height_map_depth', required=True, type=str, help='depth height map image.')
parser.add_argument('--rotation_angle', required=True, type=int, help='rotation of height map.')
parser.add_argument('--model_path', default='logs2', type=str, help='path to the trained model.')
args = parser.parse_args()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def main():
    color = cv2.imread(args.height_map_color)[..., ::-1].astype(np.float32) / 255.0
    color_backup = color
    color = (color - mean) / std
    depth = cv2.imread(args.height_map_depth, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 10000.0
    depth = np.stack([depth, depth, depth], axis=2)
    color = np.expand_dims(color, axis=0)
    depth = np.expand_dims(depth, axis=0)

    color_t = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 3])
    depth_t = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 3])

    net, _ = model(color_t,
                   depth_t,
                   num_classes=3,
                   num_channels=1000,
                   is_training=False,
                   global_pool=False,
                   output_stride=16,
                   spatial_squeeze=False,
                   color_scope='color_tower',
                   depth_scope='depth_tower',
                   scope='arcnet')
    heatmap = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    sess = tf.Session(config=session_config)
    saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
    results = sess.run(heatmap, feed_dict={color_t: color, depth_t: depth})
    results = np.squeeze(results)

    valid_mask = np.zeros((40, 40))
    valid_mask[7:33, 1:39] = 1
    valid_mask = cv2.rotate(valid_mask, 360-(45/2)*args.rotation_angle)
    affordance_map = results[..., 0]
    affordance_map[valid_mask == 0] = 0
    affordance_map = cv2.resize(affordance_map, (320, 320))
    affordance_map = cv2.GaussianBlur(affordance_map, (7, 7), 0.5)
    jet_colormap = sio.loadmat('jet_colormap.mat')['jet_colormap']
    affordance_map = jet_colormap[np.floor(affordance_map[:] * jet_colormap.shape[0]).astype(np.int)]
    result_image = ((0.5 * color_backup + 0.5 * affordance_map) * 255.0).astype(np.uint8)
    # cv2.imwrite('results.png', results.astype(np.uint8)[..., ::-1])
    cv2.imwrite('results.png', result_image[..., ::-1])
    print results.shape


if __name__ == '__main__':
    main()

'''
--height_map_color
demo/test-heightmap.color.png
--height_map_depth
demo/test-heightmap.depth.png
--rotation_angle
5
--model_path
logs2
'''

'''
--height_map_color training/color/000000-05.png --height_map_depth training/depth/000000-05.png --rotation_angle 6 --model_path logs2
'''