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

from model.GAN import GAN
from model.param import hps
from preprocess.datamining import *
from Visualize import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_path', '', 'path for training images')
tf.app.flags.DEFINE_string('label_path', '', 'path for training labels')
tf.app.flags.DEFINE_string('save_path', '', 'path for saved model')
tf.app.flags.DEFINE_string('trash_path', '', 'path for saved model')
tf.app.flags.DEFINE_string('img_save_path', '', 'path for save final image')
tf.app.flags.DEFINE_boolean('display', True, 'display image or not')
tf.app.flags.DEFINE_boolean('load', True, 'load model weights or not')
tf.app.flags.DEFINE_boolean('resize', True, 'resize imgs or not')
tf.app.flags.DEFINE_string('z_mode', 'normal', 'normal, uniform_signed, uniform_unsigned')
tf.app.flags.DEFINE_string('log_path', '/home/geonyoung/BiS800/proj5/log', 'path for logging')
tf.app.flags.DEFINE_integer('batch', 32, 'batch size')


def train(param):
    img_list = get_all_png(param.disease_group, template=param.img_path)
    print(img_list)
    if FLAGS.resize:
        trashpath = FLAGS.trash_path
        img_list = crop_resize(img_list, trashpath)
        print(img_list)
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

        epoch = param.epoch
        batch_size = param.batch_size
        num_batch_per_train_epoch = int(train_data_length / batch_size)

        summary_writer_train = tf.summary.FileWriter(FLAGS.log_path + '/train_', sess.graph)

        for i in xrange(epoch):
            sess.run(train_init_op)
            for j in xrange(num_batch_per_train_epoch):
                img_batch = sess.run(images)
                z_batch = gen_random(mode=param.z_mode, size=param.z_dim)

                # Update D network
                _, step, summary = \
                    sess.run([model.d_opt, model.global_step, model.summaries],
                             feed_dict={model.img: img_batch, model.z: z_batch})
                summary_writer_train.add_summary(summary=summary, global_step=step)

                # Update G network
                _, step, summary = \
                    sess.run([model.g_opt, model.global_step, model.summaries],
                             feed_dict={model.z: z_batch})
                summary_writer_train.add_summary(summary=summary, global_step=step)

                # Update G network once again
                _, step, summary = \
                    sess.run([model.g_opt, model.global_step, model.summaries],
                             feed_dict={model.z: z_batch})
                summary_writer_train.add_summary(summary=summary, global_step=step)

                # Eval and log loss values
                d_loss_fake = model.d_loss_fake.eval({model.z: z_batch})
                d_loss_real = model.d_loss_real.eval({model.img: img_batch, model.z: z_batch})
                g_loss = model.g_loss.eval({model.z: z_batch})

                tf.logging.info('{%d epoch, %d batch}\tg_loss: {%f} d_fake: {%f} d_real: {%f} d_total: {%f}' % (i, j, g_loss, d_loss_fake, d_loss_real, d_loss_real+d_loss_fake))

                if np.mod(step, param.save_step) == 0:
                    saver.save(sess, FLAGS.save_path, global_step=step)
                if np.mod(step, param.sample_step) == 0:
                    samples = sess.run(model.samples, feed_dict={model.z: z_batch})
                    if FLAGS.display:
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    '{}/train_{}.png'.format(FLAGS.img_save_path, step))
            tf.logging.info('{%d epoch done}' % i)


def main(_):
    param = hps
    param.img_path = FLAGS.img_path
    param.batch = FLAGS.batch
    param.display = FLAGS.display
    param.z_mode = FLAGS.z_mode

    np.random.seed(param.seed)
    train(param)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()