import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import resnet_v1


def model(colors,
          depths,
          num_classes=3,
          num_channels=1000,
          is_training=True,
          global_pool=False,
          output_stride=16,
          spatial_squeeze=False,
          color_scope='color_tower',
          depth_scope='depth_tower',
          scope='arcnet'):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        color_net, color_end_points = resnet_v1.resnet_v1_101(inputs=colors,
                                                              num_classes=num_channels,
                                                              is_training=is_training,
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              scope=color_scope)
        depth_net, depth_end_points = resnet_v1.resnet_v1_101(inputs=depths,
                                                              num_classes=num_channels,
                                                              is_training=is_training,
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              scope=depth_scope)
        net = tf.concat([color_net, depth_net], axis=3)
    with tf.variable_scope(scope, 'arcnet', [net]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # to do: add batch normalization to the following conv layers.
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_points_collection):
            net = slim.conv2d(net, 512, [1, 1], scope='conv1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv2')
            net = slim.conv2d(net, num_classes, [1, 1], scope='conv3')
            height, width = net.get_shape().as_list()[1:3]
            net = tf.image.resize_bilinear(net, [height*2, width*2], name='resize_bilinear')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    end_points.update(color_end_points)
    end_points.update(depth_end_points)
    end_points['logits'] = net
    return net, end_points


def single_tower(colors,
                 depths,
                 num_classes=3,
                 num_channels=1000,
                 is_training=True,
                 global_pool=False,
                 output_stride=16,
                 spatial_squeeze=False,
                 scope='arcnet'):
    inputs = tf.concat([colors, depths], axis=3)
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                  num_classes=num_channels,
                                                  is_training=is_training,
                                                  global_pool=global_pool,
                                                  output_stride=output_stride,
                                                  spatial_squeeze=spatial_squeeze,
                                                  scope=scope+'_tower')
    with tf.variable_scope(scope, 'arcnet', [net]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # to do: add batch normalization to the following conv layers.
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_points_collection):
            net = slim.conv2d(net, 512, [1, 1], scope='conv1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv2')
            net = slim.conv2d(net, num_classes, [1, 1], scope='conv3')
            height, width = net.get_shape().as_list()[1:3]
            net = tf.image.resize_bilinear(net, [height*2, width*2], name='resize_bilinear')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    end_points['logits'] = net
    return net, end_points


