#coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm_2d(input, name="instance_norm_2d"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(
            1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable(
            "offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def instance_norm_1d(input, name="instance_norm_1d"):
    return tf.contrib.layers.instance_norm(input, epsilon=1e-5, scale=True, scope=name)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    # padding=SAME -> (input hight or width) / strides
    # padding=VALID -> (input hight or width - filter height or width + 1) / strides .ex: (128 -7 +1)/1
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(
                               stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(
                                         stddev=stddev),
                                     biases_initializer=None)


def conv1d(input_, output_dim, input_c=1, ks=4, s=2, stddev=0.2, padding='SAME', name="conv1d"):
    with tf.variable_scope(name):
        filter = w_variable([ks, input_c, output_dim])
        return tf.nn.conv1d(input_, filters=filter, stride=s, padding=padding, use_cudnn_on_gpu=True, name=name)


def deconv1d(input_, output_shape, input_c=1, ks=4, s=2, stddev=0.2, padding='SAME', name="deconv1d"):
    with tf.variable_scope(name):
        # output_shape[1] = strides * input_[1]
        filter = w_variable([ks, output_shape[2], input_c])
        return tf.contrib.nn.conv1d_transpose(value=input_, output_shape=output_shape, filter=filter, stride=s, padding=padding, name=name)


def w_variable(shape, stddev=0.2):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.nn.leaky_relu(x, alpha=leak, name=name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
