import tensorflow as tf
import numpy as np

x = tf.placeholder(shape=[None, 256, 16], dtype=tf.float32)
# [kernel_width, output_depth, input_depth]
filter = tf.Variable(tf.random_normal([3, 8, 16]))

batch_size = tf.shape(x)[0]
x_dim = x.get_shape().as_list()

out = tf.contrib.nn.conv1d_transpose(x, filter, output_shape=[
    tf.shape(x)[0],  tf.shape(x)[1]*4, 8], stride=4, padding="SAME")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(out, feed_dict={x: np.zeros([100, 256, 16])})
    print(result.shape)
    print(sess.run(batch_size))
