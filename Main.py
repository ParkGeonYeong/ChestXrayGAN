from __future__ import print_function
import os
import tensorflow as tf

from collections import namedtuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import sys
from glob import glob
from six.moves import xrange

from tensorflow.python import debug as tf_debug
from model.GAN import GAN
from model.param import hps
from preprocess.datamining import *
from Visualize import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_path', '/home/geonyoung/BiS800/proj5/filtered_img', 'path for training images')
tf.app.flags.DEFINE_string('save_path', '/home/geonyoung/BiS800/proj5/save_model', 'path for saved model')
tf.app.flags.DEFINE_string('trash_path', '/home/geonyoung/BiS800/proj5/trash_img', 'path for bad model')
tf.app.flags.DEFINE_string('new_path', '/home/geonyoung/BiS800/proj5/resized_img', 'path for resized model')
tf.app.flags.DEFINE_string('img_save_path', '/home/geonyoung/BiS800/proj5/generated_img', 'path for save final image')
tf.app.flags.DEFINE_string('log_path', '/home/geonyoung/BiS800/proj5/log', 'path for logging')
tf.app.flags.DEFINE_string('model', 'WGAN', 'DCGAN, WGAN or WGAN-GP')
tf.app.flags.DEFINE_boolean('display', True, 'display image or not')
tf.app.flags.DEFINE_boolean('load', True, 'load model weights or not')
tf.app.flags.DEFINE_boolean('resize', False, 'resize imgs or not')
tf.app.flags.DEFINE_boolean('resize_in_sess', False, 'If do resize out of session, its more faster but have to fix size')
tf.app.flags.DEFINE_boolean('standardize', True, 'standardize imgs or not')
tf.app.flags.DEFINE_boolean('L1norm', True, 'Do L1 normalization')
tf.app.flags.DEFINE_string('z_mode', 'normal', 'normal, uniform_signed, uniform_unsigned')
tf.app.flags.DEFINE_integer('batch', 64, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 30, 'batch size')
tf.app.flags.DEFINE_integer('resize_width', 128, 'new img size')
tf.app.flags.DEFINE_integer('gf_dim', 32, 'Last feature maps of generator convolution layer')
tf.app.flags.DEFINE_integer('df_dim', 32, 'First feature maps of discriminator convolution layer')
tf.app.flags.DEFINE_integer('crop_width', 450, 'half width of a cropped image')
tf.app.flags.DEFINE_float('one_sided', 0.9, 'Adjust loss function slightly')


def train(param):
    img_list = get_all_png(param.disease_group, template=param.img_path)

    if not FLAGS.resize_in_sess:
        if FLAGS.resize:
            newpath = FLAGS.new_path
            crop_resize(img_list, crop_width=450, newpath=newpath)
        img_list = get_all_png(param.disease_group, template=FLAGS.new_path)

    train_dataset, train_data_length = load_all(img_list)

    iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types, output_shapes=train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    images = iterator.get_next()
    model = GAN(width=param.resize, height=param.resize, z_dims=param.z_dim, param=param)
    model.build_graph()

    saver = tf.train.Saver()
    init = tf.initializers.global_variables()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_l)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        epoch = param.epoch
        batch_size = param.batch_size
        num_batch_per_train_epoch = int(train_data_length / batch_size)

        summary_writer_train = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

        for i in xrange(epoch):
            sess.run(train_init_op)
            for j in xrange(num_batch_per_train_epoch):
                img_batch = sess.run(images)
                if FLAGS.resize_in_sess and FLAGS.resize:
                    img_batch = crop_resize_in_sess(img_batch, crop_width=400)
                z_batch = gen_random(mode=param.z_mode, size=(batch_size, param.z_dim))
                counter = i*num_batch_per_train_epoch+j

                d_iter = 100 if counter < 25 else param.d_iter
                # d_iter = param.d_iter
                # Update D network for d_iter
                for iter in range(d_iter):
                    tf.logging.info('{} epoch, {} batch\t{}/{} iteration'.format(i, j, iter, d_iter))
                    _, step, summary = \
                        sess.run([model.d_opt, model.global_step, model.summaries],
                                 feed_dict={model.img: img_batch, model.z: z_batch})
                    if param.model == 'WGAN':
                        sess.run(model.d_clip)
                    if d_iter != 1:
                        img_batch = sess.run(images)
                        if FLAGS.resize_in_sess and FLAGS.resize:
                            img_batch = crop_resize_in_sess(img_batch, crop_width=400)
                        z_batch = gen_random(mode=param.z_mode, size=(batch_size, param.z_dim))
                    summary_writer_train.add_summary(summary=summary, global_step=step)

                # Update G network
                _, step, summary = \
                    sess.run([model.g_opt, model.global_step, model.summaries],
                             feed_dict={model.img: img_batch, model.z: z_batch})
                summary_writer_train.add_summary(summary=summary, global_step=step)
                print(step)
                # Eval and log loss values
                d_loss = model.d_loss.eval({model.img: img_batch, model.z: z_batch})
                g_loss = model.g_loss.eval({model.z: z_batch})

                tf.logging.info('{%d epoch, %d batch}\tg_loss: {%f} d_loss: {%f}' % (i, j, g_loss, d_loss))

                if np.mod(counter, 400) == 0:
                    saver.save(sess, FLAGS.save_path, global_step=step)

                if np.mod(counter, 500) == 0:
                    samples = sess.run(model.samples, feed_dict={model.z: z_batch})
                    if FLAGS.display:
                        save_images(samples, closest_divisor(samples.shape[0]),
                                    '{}/train_{}.png'.format(FLAGS.img_save_path, counter))
            tf.logging.info('{%d epoch done}' % i)


def main(_):
    param = hps
    param.img_path = FLAGS.img_path
    param.batch_size = FLAGS.batch
    param.display = FLAGS.display
    param.z_mode = FLAGS.z_mode
    param.crop_width = FLAGS.crop_width
    param.gf_dim = FLAGS.gf_dim
    param.df_dim = FLAGS.df_dim
    param.one_sided = FLAGS.one_sided
    param.epoch = FLAGS.epoch
    param.standardize = FLAGS.standardize
    param.resize = FLAGS.resize_width
    param.model = FLAGS.model

    required_path = [FLAGS.save_path, FLAGS.img_save_path, FLAGS.log_path, FLAGS.new_path]
    make_path(required_path)
    required_path = [os.path.join(FLAGS.new_path, item.replace('No Finding', 'No_Finding')) for item in param.disease_group]
    make_path(required_path)

    np.random.seed(param.seed)
    train(param)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()