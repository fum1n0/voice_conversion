# coding: utf-8

import numpy as np
import tensorflow as tf

from tf_layer import *


def discriminator(wave, options, reuse=False, name="discriminator"):
    # wave is batch_size * fs(default=44100) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False


def generator(wave, options, reuse=False, name="generator"):
    # wave is batch_size * fs(default=44100) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # model is U_Net
        x = wave.reshape(wave.shape[0], wave.shape[1], 1)
        c1 = instance_norm_1d(conv1d(x, ouputdim=64,
                                     input_c=x.shape[2], ks=5, s=2, name='g_c1_conv1d'), name='g_c1_inorm')
