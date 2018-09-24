# coding: utf-8

import numpy as np
import tensorflow as tf

from tf_layer import *


def discriminator(wave, options, reuse=False, name="discriminator"):
    # wave is batch_size * (default=16384) * c(=1)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        wave_dim = wave.get_shape().as_list()
        # c1 is batch_size * 8192 * options.conv_dim
        c1 = lrelu(conv1d(wave, output_dim=options.conv_dim,
                          input_c=wave_dim[2], ks=15, s=2, name='d_c1_conv1d'))
        c1_dim = c1.get_shape().as_list()

        # c2 is batch_size * 4096 * (options.conv_dim*2)
        c2 = lrelu(instance_norm_1d(conv1d(c1, output_dim=options.conv_dim*2,
                                           input_c=c1_dim[2], ks=15, s=2, name='d_c2_conv1d'), name='d_c2_inorm'))
        c2_dim = c2.get_shape().as_list()

        # c3 is batch_size * 2048 * (options.conv_dim*4)
        c3 = lrelu(instance_norm_1d(conv1d(c2, output_dim=options.conv_dim*4,
                                           input_c=c2_dim[2], ks=15, s=2, name='d_c3_conv1d'), name='d_c3_inorm'))
        c3_dim = c3.get_shape().as_list()

        # c4 is batch_size * 1024 * (options.conv_dim*8)
        c4 = lrelu(instance_norm_1d(conv1d(c3, output_dim=options.conv_dim*8,
                                           input_c=c3_dim[2], ks=15, s=2, name='d_c4_conv1d'), name='d_c4_inorm'))
        c4_dim = c4.get_shape().as_list()

        # c5 is batch_size * 512 * (options.conv_dim*8)
        c5 = lrelu(instance_norm_1d(conv1d(c4, output_dim=options.conv_dim*8,
                                           input_c=c4_dim[2], ks=15, s=2, name='d_c5_conv1d'), name='d_c5_inorm'))
        c5_dim = c5.get_shape().as_list()

        # c6 is batch_size * 256 * 1
        c6 = conv1d(c5, output_dim=1,
                    input_c=c5_dim[2], ks=15, s=2, name='d_c6_conv1d')

        return c6


def generator(wave, options, reuse=False, name="generator"):
    # wave is batch_size * (default=16384) * c(=1)

    dropout_rate = 0.5 if options.is_training else 1.0

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # model is Wave U-Net
        # batch*input(16384)*c(1) -> batch*4*512 -> output
        wave_dim = wave.get_shape().as_list()
        # c1 is batch_size * 8192 * options.conv_dim
        c1 = instance_norm_1d(conv1d(wave, output_dim=options.conv_dim,
                                     input_c=wave_dim[2], ks=15, s=2, name='g_c1_conv1d'), name='g_c1_inorm')
        c1_dim = c1.get_shape().as_list()

        # c2 is batch_size * 4096 * (options.conv_dim*2)
        c2 = instance_norm_1d(conv1d(c1, output_dim=options.conv_dim*2,
                                     input_c=c1_dim[2], ks=15, s=2, name='g_c2_conv1d'), name='g_c2_inorm')
        c2_dim = c2.get_shape().as_list()

        # c3 is batch_size * 2048 * (options.conv_dim*4)
        c3 = instance_norm_1d(conv1d(c2, output_dim=options.conv_dim*4,
                                     input_c=c2_dim[2], ks=15, s=2, name='g_c3_conv1d'), name='g_c3_inorm')
        c3_dim = c3.get_shape().as_list()

        # c4 is batch_size * 1024 * (options.conv_dim*8)
        c4 = instance_norm_1d(conv1d(c3, output_dim=options.conv_dim*8,
                                     input_c=c3_dim[2], ks=15, s=2, name='g_c4_conv1d'), name='g_c4_inorm')
        c4_dim = c4.get_shape().as_list()

        # c5 is batch_size * 512 * (options.conv_dim*8)
        c5 = instance_norm_1d(conv1d(c4, output_dim=options.conv_dim*8,
                                     input_c=c4_dim[2], ks=15, s=2, name='g_c5_conv1d'), name='g_c5_inorm')
        c5_dim = c5.get_shape().as_list()

        # c6 is batch_size * 256 * (options.conv_dim*8)
        c6 = instance_norm_1d(conv1d(c5, output_dim=options.conv_dim*8,
                                     input_c=c5_dim[2], ks=15, s=2, name='g_c6_conv1d'), name='g_c6_inorm')
        c6_dim = c6.get_shape().as_list()

        # c7 is batch_size * 128 * (options.conv_dim*8)
        c7 = instance_norm_1d(conv1d(c6, output_dim=options.conv_dim*8,
                                     input_c=c6_dim[2], ks=15, s=2, name='g_c7_conv1d'), name='g_c7_inorm')
        c7_dim = c7.get_shape().as_list()

        # c8 is batch_size * 64 * (options.conv_dim*8)
        c8 = instance_norm_1d(conv1d(c7, output_dim=options.conv_dim*8,
                                     input_c=c7_dim[2], ks=15, s=2, name='g_c8_conv1d'), name='g_c8_inorm')
        c8_dim = c8.get_shape().as_list()

        # c9 is batch_size * 32 * (options.conv_dim*8)
        c9 = instance_norm_1d(conv1d(c8, output_dim=options.conv_dim*8,
                                     input_c=c8_dim[2], ks=15, s=2, name='g_c9_conv1d'), name='g_c9_inorm')
        c9_dim = c9.get_shape().as_list()

        # c10 is batch_size * 16 * (options.conv_dim*8)
        c10 = instance_norm_1d(conv1d(c9, output_dim=options.conv_dim*8,
                                      input_c=c9_dim[2], ks=15, s=2, name='g_c10_conv1d'), name='g_c10_inorm')
        c10_dim = c10.get_shape().as_list()

        # c11 is batch_size * 8 * (options.conv_dim*8)
        c11 = instance_norm_1d(conv1d(c10, output_dim=options.conv_dim*8,
                                      input_c=c10_dim[2], ks=15, s=2, name='g_c11_conv1d'), name='g_c11_inorm')
        c11_dim = c11.get_shape().as_list()

        # c12 is batch_size * 4 * (options.conv_dim*8)
        c12 = instance_norm_1d(conv1d(c11, output_dim=options.conv_dim*8,
                                      input_c=c11_dim[2], ks=15, s=2, name='g_c12_conv1d'), name='g_c12_inorm')
        c12_dim = c12.get_shape().as_list()

        # d1 is batch_size * 8 * (options.conv_dim*8*2)
        d1 = deconv1d(c12, output_shape=[tf.shape(c12)[0], (int)(c12_dim[1] * 2),
                                         (int)(c12_dim[2])], input_c=c12_dim[2], ks=5, s=2, name='g_d1_deconv1d')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm_1d(d1, name='g_d1_inorm'), c11], 2)
        d1_dim = d1.get_shape().as_list()

        # d2 is batch_size * 16 * (options.conv_dim*8*2)
        d2 = deconv1d(d1, output_shape=[tf.shape(d1)[0], (int)(d1_dim[1] * 2),
                                        (int)(d1_dim[2]/2)], input_c=d1_dim[2], ks=5, s=2, name='g_d2_deconv1d')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm_1d(d2, name='g_d2_inorm'), c10], 2)
        d2_dim = d2.get_shape().as_list()

        # d3 is batch_size * 32 * (options.conv_dim*8*2)
        d3 = deconv1d(d2, output_shape=[tf.shape(d2)[0], (int)(d2_dim[1] * 2),
                                        (int)(d2_dim[2]/2)], input_c=d2_dim[2], ks=5, s=2, name='g_d3_deconv1d')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm_1d(d3, name='g_d3_inorm'), c9], 2)
        d3_dim = d3.get_shape().as_list()

        # d4 is batch_size * 64 * (options.conv_dim*8*2)
        d4 = deconv1d(d3, output_shape=[tf.shape(d3)[0], (int)(d3_dim[1] * 2),
                                        (int)(d3_dim[2]/2)], input_c=d3_dim[2], ks=5, s=2, name='g_d4_deconv1d')
        d4 = tf.nn.dropout(d4, dropout_rate)
        d4 = tf.concat([instance_norm_1d(d4, name='g_d4_inorm'), c8], 2)
        d4_dim = d4.get_shape().as_list()

        # d5 is batch_size * 128 * (options.conv_dim*8*2)
        d5 = deconv1d(d4, output_shape=[tf.shape(d4)[0], (int)(d4_dim[1] * 2),
                                        (int)(d4_dim[2]/2)], input_c=d4_dim[2], ks=5, s=2, name='g_d5_deconv1d')
        d5 = tf.concat([instance_norm_1d(d5, name='g_d5_inorm'), c7], 2)
        d5_dim = d5.get_shape().as_list()

        # d6 is batch_size * 256 * (options.conv_dim*8*2)
        d6 = deconv1d(d5, output_shape=[tf.shape(d5)[0], (int)(d5_dim[1] * 2),
                                        (int)(d5_dim[2]/2)], input_c=d5_dim[2], ks=5, s=2, name='g_d6_deconv1d')
        d6 = tf.concat([instance_norm_1d(d6, name='g_d6_inorm'), c6], 2)
        d6_dim = d6.get_shape().as_list()

        # d7 is batch_size * 512 * (options.conv_dim*8*2)
        d7 = deconv1d(d6, output_shape=[tf.shape(d6)[0], (int)(d6_dim[1] * 2),
                                        (int)(d6_dim[2]/2)], input_c=d6_dim[2], ks=5, s=2, name='g_d7_deconv1d')
        d7 = tf.concat([instance_norm_1d(d7, name='g_d7_inorm'), c5], 2)
        d7_dim = d7.get_shape().as_list()

        # d8 is batch_size * 1024 * (options.conv_dim*8*2)
        d8 = deconv1d(d7, output_shape=[tf.shape(d7)[0], (int)(d7_dim[1] * 2),
                                        (int)(d7_dim[2]/2)], input_c=d7_dim[2], ks=5, s=2, name='g_d8_deconv1d')
        d8 = tf.concat([instance_norm_1d(d8, name='g_d8_inorm'), c4], 2)
        d8_dim = d8.get_shape().as_list()

        # d9 is batch_size * 2048 * (options.conv_dim*4*2)
        d9 = deconv1d(d8, output_shape=[tf.shape(d8)[0], (int)(d8_dim[1] * 2),
                                        (int)(d8_dim[2]/(2*2))], input_c=d8_dim[2], ks=5, s=2, name='g_d9_deconv1d')
        d9 = tf.concat([instance_norm_1d(d9, name='g_d9_inorm'), c3], 2)
        d9_dim = d9.get_shape().as_list()

        # d10 is batch_size * 4096 * (options.conv_dim*2*2)
        d10 = deconv1d(d9, output_shape=[tf.shape(d9)[0], (int)(d9_dim[1] * 2),
                                         (int)(d9_dim[2]/(2*2))], input_c=d9_dim[2], ks=5, s=2, name='g_d10_deconv1d')
        d10 = tf.concat([instance_norm_1d(d10, name='g_d10_inorm'), c2], 2)
        d10_dim = d10.get_shape().as_list()

        # d11 is batch_size * 8192 * (options.conv_dim*1*2)
        d11 = deconv1d(d10, output_shape=[tf.shape(d10)[0], (int)(d10_dim[1] * 2),
                                          (int)(d10_dim[2]/(2*2))], input_c=d10_dim[2], ks=5, s=2, name='g_d11_deconv1d')
        d11 = tf.concat([instance_norm_1d(d11, name='g_d11_inorm'), c1], 2)
        d11_dim = d11.get_shape().as_list()

        # d12 is batch_size * 16384 * (1*2)
        d12 = deconv1d(d11, output_shape=[tf.shape(d11)[0], (int)(d11_dim[1] * 2),
                                          1], input_c=d11_dim[2], ks=5, s=2, name='g_d12_deconv1d')
        d12 = tf.concat([instance_norm_1d(d12, name='g_d12_inorm'), wave], 2)
        d12_dim = d12.get_shape().as_list()

        # d13 is batch_size * 16384 * 1
        d13 = deconv1d(d12, output_shape=[tf.shape(d12)[0], (int)(
            d12_dim[1]), 1], input_c=d12_dim[2], ks=1, s=1, name='g_d13_deconv1d')

        return tf.nn.tanh(d13)
