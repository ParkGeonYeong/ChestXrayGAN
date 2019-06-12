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
        self.model = param.model
        self.gp_lambda = param.gp_lambda
        self.L1norm = param.L1norm

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')


        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')

    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope("Preprocessing"):
            if self.param.standardize:
                img = self.preprocess(self.img)
            else:
                img = self.img
            z = self.z
        init_stddev = 0.075
        # Should Control its step
        stddev = tf.train.exponential_decay(init_stddev, self.global_step, 1000, 0.95, staircase=True)
        D_gaussian_noise = tf.random.normal(tf.shape(img), mean=0., stddev=stddev)

        # Add decaying gaussian noise
        self.G = self.Generator(z)
        self.D, self.D_logits = self.Discriminator(img+D_gaussian_noise, reuse=False)
        self.D_fake, self.D_fake_logits = self.Discriminator(self.G+D_gaussian_noise, reuse=True)

        if self.model == 'GAN':
            self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.constant(self.param.one_sided)*tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake)))
            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.ones_like(self.D_fake)))
        else:
            # W-GAN is not about cross entropy
            self.d_loss_real = tf.reduce_mean(self.D_logits)
            self.d_loss_fake = tf.reduce_mean(self.D_fake_logits)
            self.d_loss = self.d_loss_fake - self.d_loss_real
            self.g_loss = -tf.reduce_mean(self.D_fake_logits)

            if self.model == 'WGAN-GP':
                eps = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated_x = eps * img + (1-eps) * self.G

                # Now constraints gradient of this interpolated new data to 1
                _, self.D_inter_logits = self.Discriminator(interpolated_x, reuse=True)
                gradients = tf.gradients(self.D_inter_logits, [interpolated_x, ], name="D_logits_intp")[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))
                self.d_loss += self.gp_lambda * grad_penalty
                tf.summary.scalar('discriminator/penalty_loss', grad_penalty)
                tf.summary.scalar('discriminator/grad_norm', tf.nn.l2_loss(gradients))

        if self.L1norm:
            pass

        with tf.variable_scope("Seperate_trainable_vars"):
            trainable_vars = tf.trainable_variables()
            self.d_vars = [var for var in trainable_vars if 'd_' in var.name]
            self.g_vars = [var for var in trainable_vars if 'g_' in var.name]

        with tf.variable_scope("Training"):
            if self.model == 'GAN':
                print('i am not in GAN')
                self.d_opt = tf.train.AdamOptimizer(self.param.d_lr, beta1=self.param.beta1).minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
                self.g_opt = tf.train.AdamOptimizer(self.param.g_lr, beta1=self.param.beta1).minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)
            elif self.model == 'WGAN':
                self.d_opt = tf.train.RMSPropOptimizer(self.param.d_lr).minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
                self.d_clip = tf.group(*[v.assign(tf.clip_by_value(v, -self.param.w_clip, self.param.w_clip)) for v in self.d_vars])
                self.g_opt = tf.train.RMSPropOptimizer(self.param.g_lr).minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)
            elif self.model == 'WGAN-GP':
                self.d_opt = tf.train.AdamOptimizer(self.param.d_lr, beta1=self.param.beta1, beta2=self.param.beta2).minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
                self.g_opt = tf.train.AdamOptimizer(self.param.g_lr, beta1=self.param.beta1, beta2=self.param.beta2).minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)

        tf.summary.scalar('discriminator/real_loss', self.d_loss_real)
        tf.summary.scalar('discriminator/fake_loss', self.d_loss_fake)
        tf.summary.scalar('discriminator/total_loss', self.d_loss)
        tf.summary.scalar('generator/loss', self.g_loss)
        tf.summary.image('generator/images', self.G, max_outputs=4)
        tf.summary.image('generator/original_images', img, max_outputs=4)

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
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            z_ = linear(z, self.gf_dim * 32 * s_h64 * s_w64, name='g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h64, s_w64, self.gf_dim*32])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv(h0, self.gf_dim * 32, s_h32, s_w32, self.gf_dim * 16, kernel=4, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv(h1, self.gf_dim * 16, s_h16, s_w16, self.gf_dim * 8, kernel=4, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv(h2, self.gf_dim*8, s_h8, s_w8, self.gf_dim*4, kernel=4, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv(h3, self.gf_dim*4, s_h4, s_w4, self.gf_dim*2, kernel=4, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4))

            h5 = deconv(h4, self.gf_dim*2, s_h2, s_w2, self.gf_dim*1, kernel=4, name='g_h5')
            h5 = tf.nn.relu(self.g_bn5(h5))

            h6 = deconv(h5, self.gf_dim*1, s_h, s_w, self.c_dim, kernel=4, name='g_h6')
            # Skip last batch normalization

            return tf.nn.tanh(h6)

    def Sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.height, self.width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            z_ = linear(z, self.gf_dim * 32 * s_h64 * s_w64, name='g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h64, s_w64, self.gf_dim * 32])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv(h0, self.gf_dim * 32, s_h32, s_w32, self.gf_dim * 16, kernel=4, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv(h1, self.gf_dim * 16, s_h16, s_w16, self.gf_dim * 8, kernel=4, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv(h2, self.gf_dim * 8, s_h8, s_w8, self.gf_dim * 4, kernel=4, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv(h3, self.gf_dim * 4, s_h4, s_w4, self.gf_dim * 2, kernel=4, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4))

            h5 = deconv(h4, self.gf_dim * 2, s_h2, s_w2, self.gf_dim * 1, kernel=4, name='g_h5')
            h5 = tf.nn.relu(self.g_bn5(h5))

            h6 = deconv(h5, self.gf_dim * 1, s_h, s_w, self.c_dim, kernel=4, name='g_h6')
            # Skip last batch normalization

            return tf.nn.tanh(h6)

    def Discriminator(self, input, reuse=True):
        with tf.variable_scope("Discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv(input, in_filters=self.c_dim, kernel_size=5, out_filters=self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv(h0, in_filters=self.df_dim, kernel_size=5, out_filters=self.df_dim*2, name='d_h1_conv')))
            h1 = tf.layers.dropout(h1, rate=0.5, seed=self.param.seed, name='d_h1_dropout')

            h2 = lrelu(self.d_bn2(conv(h1, in_filters=self.df_dim*2, kernel_size=5, out_filters=self.df_dim*4, name='d_h2_conv')))
            h2 = tf.layers.dropout(h2, rate=0.5, seed=self.param.seed, name='d_h2_dropout')

            h3 = lrelu(self.d_bn3(conv(h2, in_filters=self.df_dim*4, kernel_size=5, out_filters=self.df_dim*8, name='d_h3_conv')))
            h3 = tf.layers.dropout(h3, rate=0.5, seed=self.param.seed, name='d_h3_dropout')

            h4 = lrelu(self.d_bn4(
                conv(h3, in_filters=self.df_dim*8, kernel_size=5, out_filters=self.df_dim * 16, name='d_h4_conv')))
            h4 = tf.layers.dropout(h4, rate=0.5, seed=self.param.seed, name='d_h4_dropout')

            h5 = lrelu(self.d_bn5(
                conv(h4, in_filters=self.df_dim*16, kernel_size=5, out_filters=self.df_dim * 32, name='d_h5_conv')))
            h5 = tf.layers.dropout(h5, rate=0.5, seed=self.param.seed, name='d_h5_dropout')

            h6 = linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_h0_lin')
            return tf.nn.sigmoid(h6), h6
