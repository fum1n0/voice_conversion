# coding: utf-8

import numpy as np
import tensorflow as tf

from tf_layer import *


def discriminator(wave, reuse=False, name="discriminator"):
    # wave is batch_size * fs(default=44100) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False


def generator(wave, reuse=False, name="generator"):
    # wave is batch_size * fs(default=44100) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        cdim = 2

        # model is U_Net
        x = wave.reshape(wave.shape[0], wave.shape[1], 1)

        # c1 is batch_size * 22500 * cdim
        c1 = instance_norm_1d(conv1d(x, output_dim=cdim,
                                     input_c=x.shape[2], ks=4, s=2, name='g_c1_conv1d'), name='g_c1_inorm')
        c1_dim = c1.get_shape().as_list()

        # c2 is batch_size * 11250 * (cdim*2)
        c2 = instance_norm_1d(conv1d(c1, output_dim=cdim*2,
                                     input_c=c1_dim[2], ks=4, s=2, name='g_c2_conv1d'), name='g_c2_inorm')
        c2_dim = c2.get_shape().as_list()

        # d1 is batch_size * 22500 * cdim
        d1 = instance_norm_1d(
            deconv1d(c2, output_shape=[c2_dim[0], (int)(c2_dim[1] * 2),
                                       (int)(c2_dim[2]/2)], input_c=c2_dim[2], ks=4, s=2, name='g_d1_deconv1d'), name='g_d1_inorm')
        d1_dim = d1.get_shape().as_list()

        # d2 is batch_size * 44100 * 1
        d2 = instance_norm_1d(
            deconv1d(d1, output_shape=[d1_dim[0], (int)(d1_dim[1] * 2),
                                       (int)(d1_dim[2]/2)], input_c=d1_dim[2], ks=4, s=2, name='g_d2_deconv1d'), name='g_d2_inorm')

        return tf.nn.tanh(d2)


if __name__ == "__main__":
    x = np.empty((0, 44100), np.float32)
    t1 = np.ones([44100])
    t2 = np.zeros([44100])
    x = np.append(x, np.array([t1]), axis=0)
    x = np.append(x, np.array([t2]), axis=0)

    x = x.reshape(x.shape[0], x.shape[1], 1).astype(np.float32)

    y = generator(x)
    y_dim = y.get_shape().as_list()
    z = tf.reshape(y, [y_dim[0], y_dim[1]])

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(z))
    print(y_dim)
