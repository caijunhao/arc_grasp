import tensorflow as tf


def create_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.001,
                                          lr_decay_step=100000,
                                          lr_decay_rate=0.97,
                                          lamb=15.0,
                                          batch_size=8,
                                          image_size=320,
                                          model_name='resnet_v1_101')
    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams
