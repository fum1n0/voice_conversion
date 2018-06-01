import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv1d(x, W, padding='SAME'):
    return tf.nn.conv1d(x, W, stride=1, padding=padding)


i = tf.constant([1, 0, 2, 3, 0, 1, 1], dtype=tf.float32, name='i')

data = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
#data = tf.pad(data, [[0, 0], [1, 1], [0, 0]])

sess = tf.Session()

W_conv = weight_variable([3, 1, 1])
b_conv = bias_variable([1, 1])
init = tf.initialize_all_variables()
sess.run(init)
print(sess.run(W_conv))

k = tf.constant([2, 1, 3], dtype=tf.float32, name='k')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
res = conv1d(data, kernel, padding='VALID')
print(sess.run(kernel))
print(sess.run(res))

output = conv1d(data, W_conv)
output = tf.nn.relu(output + b_conv)
output = tf.reshape(output, [int(output.shape[1])])

print(sess.run(output))
