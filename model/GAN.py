from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import math
from collections import namedtuple
from model.layers import *


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class GAN:
    def __init__(self, width, height, z_dims, param):
        z = tf.placeholder(dtype=tf.float32, shape=(param.batch_size, z_dims))
        img = tf.placeholder(dtype=tf.float32, shape=(param.batch_size, width, height, 1))

        self.z = z
        self.img = img
        self.height = height
        self.width = width
        self.param = param
        self.batch_size = param.batch_size
        self.gf_dim = param.gf_dim # Generator's Last feature dimensions
        self.df_dim = param.df_dim
        self.c_dim = param.num_classes # Gray-scale image

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope("Preprocessing"):
            img = self.preprocess(self.img)
            z = self.z
        self.G = self.Generator(z)
        self.D, self.D_logits = self.Discriminator(img, reuse=False)
        self.D_fake, self.D_fake_logits = self.Discriminator(self.G, reuse=True)

        with tf.variable_scope("Define_minmax_loss"):
            self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake)))
            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.ones_like(self.D_fake)))

        with tf.variable_scope("Seperate_trainable_vars"):
            trainable_vars = tf.trainable_variables()
            self.d_vars = [var for var in trainable_vars if 'd_' in var.name]
            self.g_vars = [var for var in trainable_vars if 'g_' in var.name]

        with tf.variable_scope("Training"):
            self.d_opt = tf.train.AdamOptimizer(self.param.lr, beta1=self.param.beta1).minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
            self.g_opt = tf.train.AdamOptimizer(self.param.lr, beta1=self.param.beta1).minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)

        tf.summary.scalar('discriminator/real_loss', self.d_loss_real)
        tf.summary.scalar('discriminator/fake_loss', self.d_loss_fake)
        tf.summary.scalar('discriminator/total_loss', self.d_loss)
        tf.summary.scalar('generator/loss', self.g_loss)
        self.samples = self.Sampler(self.z)

    def preprocess(self, img):
        img_ = tf.map_fn(lambda each_frame: tf.image.per_image_standardization(each_frame), img)
        return img_

    def Generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.height, self.width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, name='g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv(h0, self.gf_dim*8, s_h8, s_w8, self.gf_dim*4, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv(h1, self.gf_dim*4, s_h4, s_w4, self.gf_dim*2, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv(h2, self.gf_dim*2, s_h2, s_w2, self.gf_dim*1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv(h3, self.gf_dim*1, s_h, s_w, self.c_dim, name='g_h4')
            # Skip last batch normalization

            return tf.nn.tanh(h4)

    def Sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.height, self.width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, name='g_h0_lin')

            h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv(h0, self.gf_dim*8, s_h8, s_w8, self.gf_dim*4, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv(h1, self.gf_dim*4, s_h4, s_w4, self.gf_dim*2, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv(h2, self.gf_dim*2, s_h2, s_w2, self.gf_dim*1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv(h3, self.gf_dim*1, s_h, s_w, self.c_dim, name='g_h4')
            # Skip last batch normalization

            return tf.nn.tanh(h4)

    def Discriminator(self, input, reuse=True):
        with tf.variable_scope("Discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv(input, in_filters=self.c_dim, kernel_size=5, out_filters=self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv(h0, in_filters=self.df_dim, kernel_size=5, out_filters=self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv(h1, in_filters=self.df_dim*2, kernel_size=5, out_filters=self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv(h2, in_filters=self.df_dim*4, kernel_size=5, out_filters=self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4), h4
