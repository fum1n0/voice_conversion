# coding: utf-8

import numpy as np
import tensorflow as tf

from tf_layer import *


def discriminator(wave, options, reuse=False, name="discriminator"):
    # wave is batch_size * (default=32768) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        x = wave.reshape(wave.shape[0], wave.shape[1], 1)

        # c1 is batch_size * 16384 * options.conv_dim
        c1 = lrelu(instance_norm_1d(conv1d(x, output_dim=options.conv_dim,
                                           input_c=x.shape[2], ks=8, s=2, name='d_c1_conv1d'), name='d_c1_inorm'))
        c1_dim = c1.get_shape().as_list()

        # c2 is batch_size * 8192 * (options.conv_dim*2)
        c2 = lrelu(instance_norm_1d(conv1d(c1, output_dim=options.conv_dim*2,
                                           input_c=c1_dim[2], ks=8, s=2, name='d_c2_conv1d'), name='d_c2_inorm'))
        c2_dim = c2.get_shape().as_list()

        # c3 is batch_size * 4096 * (options.conv_dim*4)
        c3 = lrelu(instance_norm_1d(conv1d(c2, output_dim=options.conv_dim*4,
                                           input_c=c2_dim[2], ks=8, s=2, name='d_c3_conv1d'), name='d_c3_inorm'))
        c3_dim = c3.get_shape().as_list()

        # c4 is batch_size * 2048 * (options.conv_dim*8)
        c4 = lrelu(instance_norm_1d(conv1d(c3, output_dim=options.conv_dim*8,
                                           input_c=c3_dim[2], ks=8, s=2, name='d_c4_conv1d'), name='d_c4_inorm'))
        c4_dim = c4.get_shape().as_list()

        # c5 is batch_size * 1024 * (options.conv_dim*8)
        c5 = lrelu(instance_norm_1d(conv1d(c4, output_dim=options.conv_dim*8,
                                           input_c=c4_dim[2], ks=8, s=2, name='d_c5_conv1d'), name='d_c5_inorm'))
        c5_dim = c5.get_shape().as_list()

        # c6 is batch_size * 512 * (options.conv_dim*8)
        c6 = lrelu(instance_norm_1d(conv1d(c5, output_dim=options.conv_dim*8,
                                           input_c=c5_dim[2], ks=8, s=2, name='d_c6_conv1d'), name='d_c6_inorm'))
        c6_dim = c6.get_shape().as_list()

        # c7 is batch_size * 256 * (options.conv_dim*8)
        c7 = lrelu(instance_norm_1d(conv1d(c6, output_dim=options.conv_dim*8,
                                           input_c=c6_dim[2], ks=8, s=2, name='d_d7_conv1d'), name='d_d7_inorm'))
        c7_dim = c7.get_shape().as_list()

        # c8 is batch_size * 128 * (options.conv_dim*8)
        c8 = lrelu(instance_norm_1d(conv1d(c7, output_dim=options.conv_dim*8,
                                           input_c=c7_dim[2], ks=8, s=2, name='d_c8_conv1d'), name='d_c8_inorm'))
        c8_dim = c8.get_shape().as_list()

        # c9 is batch_size * 128 * 1
        c9 = conv1d(c8, output_dim=1,
                    input_c=c8_dim[2], ks=8, s=1, name='d_c9_conv1d')

        return c9


def generator(wave, options, reuse=False, name="generator"):
    # wave is batch_size * (default=32768) * c(=1)

    dropout_rate = 0.5 if options.is_traning else 1.0

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # model is U_Net
        x = wave.reshape(wave.shape[0], wave.shape[1], 1)

        # input -> input / (2^9) -> output

        # c1 is batch_size * 16384 * options.conv_dim
        c1 = instance_norm_1d(conv1d(x, output_dim=options.conv_dim,
                                     input_c=x.shape[2], ks=8, s=2, name='g_c1_conv1d'), name='g_c1_inorm')
        c1_dim = c1.get_shape().as_list()

        # c2 is batch_size * 8192 * (options.conv_dim*2)
        c2 = instance_norm_1d(conv1d(c1, output_dim=options.conv_dim*2,
                                     input_c=c1_dim[2], ks=8, s=2, name='g_c2_conv1d'), name='g_c2_inorm')
        c2_dim = c2.get_shape().as_list()

        # c3 is batch_size * 4096 * (options.conv_dim*4)
        c3 = instance_norm_1d(conv1d(c2, output_dim=options.conv_dim*4,
                                     input_c=c2_dim[2], ks=8, s=2, name='g_c3_conv1d'), name='g_c3_inorm')
        c3_dim = c3.get_shape().as_list()

        # c4 is batch_size * 2048 * (options.conv_dim*8)
        c4 = instance_norm_1d(conv1d(c3, output_dim=options.conv_dim*8,
                                     input_c=c3_dim[2], ks=8, s=2, name='g_c4_conv1d'), name='g_c4_inorm')
        c4_dim = c4.get_shape().as_list()

        # c5 is batch_size * 1024 * (options.conv_dim*8)
        c5 = instance_norm_1d(conv1d(c4, output_dim=options.conv_dim*8,
                                     input_c=c4_dim[2], ks=8, s=2, name='g_c5_conv1d'), name='g_c5_inorm')
        c5_dim = c5.get_shape().as_list()

        # c6 is batch_size * 512 * (options.conv_dim*8)
        c6 = instance_norm_1d(conv1d(c5, output_dim=options.conv_dim*8,
                                     input_c=c5_dim[2], ks=8, s=2, name='g_c6_conv1d'), name='g_c6_inorm')
        c6_dim = c6.get_shape().as_list()

        # c7 is batch_size * 256 * (options.conv_dim*8)
        c7 = instance_norm_1d(conv1d(c6, output_dim=options.conv_dim*8,
                                     input_c=c6_dim[2], ks=8, s=2, name='g_c7_conv1d'), name='g_c7_inorm')
        c7_dim = c7.get_shape().as_list()

        # c8 is batch_size * 128 * (options.conv_dim*8)
        c8 = instance_norm_1d(conv1d(c7, output_dim=options.conv_dim*8,
                                     input_c=c7_dim[2], ks=8, s=2, name='g_c8_conv1d'), name='g_c8_inorm')
        c8_dim = c8.get_shape().as_list()

        # c9 is batch_size * 64 * (options.conv_dim*8)
        c9 = instance_norm_1d(conv1d(c8, output_dim=options.conv_dim*8,
                                     input_c=c8_dim[2], ks=8, s=2, name='g_c9_conv1d'), name='g_c9_inorm')
        c9_dim = c9.get_shape().as_list()

        # d1 is batch_size * 128 * (options.conv_dim*8*2)
        d1 = deconv1d(c9, output_shape=[c9_dim[0], (int)(c9_dim[1] * 2),
                                        (int)(c9_dim[2])], input_c=c9_dim[2], ks=8, s=2, name='g_d1_deconv1d')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm_1d(d1, name='g_d1_inorm'), c8], 2)
        d1_dim = d1.get_shape().as_list()

        # d2 is batch_size * 256 * (options.conv_dim*8*2)
        d2 = deconv1d(d1, output_shape=[d1_dim[0], (int)(d1_dim[1] * 2),
                                        (int)(d1_dim[2]/2)], input_c=d1_dim[2], ks=8, s=2, name='g_d2_deconv1d')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm_1d(d2, name='g_d2_inorm'), c7], 2)
        d2_dim = d2.get_shape().as_list()

        # d3 is batch_size * 512 * (options.conv_dim*8*2)
        d3 = deconv1d(d2, output_shape=[d2_dim[0], (int)(d2_dim[1] * 2),
                                        (int)(d2_dim[2]/2)], input_c=d2_dim[2], ks=8, s=2, name='g_d3_deconv1d')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm_1d(d3, name='g_d3_inorm'), c6], 2)
        d3_dim = d3.get_shape().as_list()

        # d4 is batch_size * 1024 * (options.conv_dim*8*2)
        d4 = deconv1d(d3, output_shape=[d3_dim[0], (int)(d3_dim[1] * 2),
                                        (int)(d3_dim[2]/2)], input_c=d3_dim[2], ks=8, s=2, name='g_d4_deconv1d')
        d4 = tf.nn.dropout(d4, dropout_rate)
        d4 = tf.concat([instance_norm_1d(d4, name='g_d4_inorm'), c5], 2)
        d4_dim = d4.get_shape().as_list()

        # d5 is batch_size * 2048 * (options.conv_dim*8*2)
        d5 = deconv1d(d4, output_shape=[d4_dim[0], (int)(d4_dim[1] * 2),
                                        (int)(d4_dim[2]/2)], input_c=d4_dim[2], ks=8, s=2, name='g_d5_deconv1d')
        d5 = tf.concat([instance_norm_1d(d5, name='g_d5_inorm'), c4], 2)
        d5_dim = d5.get_shape().as_list()

        # d6 is batch_size * 4096 * (options.conv_dim*4*2)
        d6 = deconv1d(d5, output_shape=[d5_dim[0], (int)(d5_dim[1] * 2),
                                        (int)(d5_dim[2]/(2*2))], input_c=d5_dim[2], ks=8, s=2, name='g_d6_deconv1d')
        d6 = tf.concat([instance_norm_1d(d6, name='g_d6_inorm'), c3], 2)
        d6_dim = d6.get_shape().as_list()

        # d7 is batch_size * 8192 * (options.conv_dim*2*2)
        d7 = deconv1d(d6, output_shape=[d6_dim[0], (int)(d6_dim[1] * 2),
                                        (int)(d6_dim[2]/(2*2))], input_c=d6_dim[2], ks=8, s=2, name='g_d7_deconv1d')
        d7 = tf.concat([instance_norm_1d(d7, name='g_d7_inorm'), c2], 2)
        d7_dim = d7.get_shape().as_list()

        # d8 is batch_size * 16384 * (options.conv_dim*1*2)
        d8 = deconv1d(d7, output_shape=[d7_dim[0], (int)(d7_dim[1] * 2),
                                        (int)(d7_dim[2]/(2*2))], input_c=d7_dim[2], ks=8, s=2, name='g_d8_deconv1d')
        d8 = tf.concat([instance_norm_1d(d8, name='g_d8_inorm'), c1], 2)
        d8_dim = d8.get_shape().as_list()

        # d9 is batch_size * 32768 * 1
        d9 = deconv1d(d8, output_shape=[d8_dim[0], (int)(
            d8_dim[1] * 2), 1], input_c=d8_dim[2], ks=8, s=2, name='g_d9_deconv1d')

        return tf.nn.tanh(d9)