import math
import numpy as np
import tensorflow as tf


def deconv(x, w, h, out_filters, stride=2, name="deconv"):
    with tf.variable_scope(name):
        x_size = tf.shape(x)[1]
        y_size = tf.shape(x)[2]
        in_filters = tf.shape(x)[-1]
        batch = tf.shape(x)[0]

        output_shape = tf.stack([batch, 2 * x_size, 2 * y_size, out_filters])
        N = in_filters * w * h

        kernel = tf.get_variable('deconv_kernel', shape=[w, h, out_filters, in_filters],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=tf.sqrt(2 / N)))
    return tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, stride, stride, 1])


def conv(x, kernel_size, out_filters, stride=2, padding='SAME', name="conv"):
    with tf.variable_scope(name):
        in_filters = tf.shape(x)[-1]
        N = in_filters * kernel_size * kernel_size
        kernel = tf.get_variable('kernel', shape=[kernel_size, kernel_size, in_filters, out_filters],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2 / N)))
        x = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=padding)
    return x


def maxpool(x, k, stride, name="maxpool"):
    with tf.variable_scope(name):
        ksize = [1, k, k, 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding='SAME')


def lrelu(x, leak=0.2, name="lelu"):
  return tf.maximum(x, leak*x)


def linear(input_, output_size, name="Linear", stddev=0.02, bias_start=0.0):
  # usually 1-D
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    try:
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    except ValueError as err:
        msg = "NOTE: Error in linear weight initialization. Usually, this is due to an dimensional issue"
        err.args = err.args + (msg,)
        raise
    bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

  return tf.matmul(input_, matrix) + bias


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)