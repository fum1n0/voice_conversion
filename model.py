# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import time
import csv
import shutil
from glob import glob
from collections import namedtiple

from module import *
from utils import *


class CycleGAN(object):

    def __init__(self, sess, args):

        self.sess = sess
        self.batch_size = args.batch_size
        self.sig_len = args.sig_len
        self.fl = args.fl
        self.fp = args.fp
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = self.dataset_dir

        self.discriminator = discriminator
        self.generator = generator

        OPTIONS = namedtuple(
            'OPTIONS', 'batch_size sig_len conv_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.sig_len, args.conv_dim, args.phase='train'))

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        # generator
        self.real_data = tf.placeholder(
            tf.float32, [None, self.sig_len, 2], name='real_A_and_B_signal')
        self.real_A = self.real_data[:, :, :1]
        self.real_B = self.real_data[:, :, 1:2]

        self.fake_B = self.generator(
            self.real_A, self.options, False, name='generatorA2B')  # Fake B
        self.fake_A_ = self.genetator(
            self.fake_B, self.options, False, name='generatorB2A')  # Cycle Consistency A

        self.fake_A = self.generator(
            self.real_B, self.options, True, name='generatorB2A')  # Fake A
        self.fake_B_ = self.generator(
            self.fake_A, self.options, True, name='generatorA2B')  # Cycle Consistency B

        self.DB_fake = self.discriminator(
            self.fake_B, self.options, False, name='discriminatorB')
        self.DA_fake = self.discriminator(
            self.fake_A, self.options, False, name='discriminatorA')

        self.L1_lambda_tf = tf.placeholder(tf.float32)

        self.g_cyc_loss = self.L1_lambda_tf * \
            tf.reduce_mean(tf.abs(self.real_A - self.fake_A_))
            + self.L1_lambda_tf * \
                tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
        self.g_loss = tf.reduce_mean(tf.abs(self.DA_fake - tf.ones_like(self.DA_fake)))  \
            + tf.reduce_mean(tf.abs(self.DB_fake - tf.ones_like(self.DB_fake)))
            + self.g_cyc_loss

        # discriminator
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.sig_len, 1], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.sig_len, 1], name='fake_B_sample')
        self.DB_real = self.discriminator(
            self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(
            self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(
            self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(
            self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.DB_res = self.DB_real + self.DB_fake
        self.DA_res = self.DA_real + self.DA_fake

        self.db_loss_real = tf.reduce_mean(
            tf.abs(self.DB_real - tf.ones_like(self.DB_real)))
        self.db_loss_fake = tf.reduce_mean(
            tf.abs(self.DB_fake_sample - tf.zeros_like(self.DB_fake_sample)))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

        self.da_loss_real = tf.reduce_mean(
            tf.abs(self.DA_real - tf.ones_like(self.DA_real)))
        self.da_loss_fake = tf.reduce_mean(
            tf.abs(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample)))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

        self.d_loss = self.da_loss + self.db_loss

        # summary
        self.L1_lambda_sum = tf.summary.scalar(
            "L1_lambda", self.L1_lambda_tf)

        self.g_cyc_loss_sum = tf.summary.scalar("g_cyc_loss", self.g_cyc_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.g_sum = tf.summary.merge(
            [self.L1_lambda_sum, self.g_cyc_loss_sum, self.g_loss_sum])

        self.db_real_sum = tf.summary.scalar(
            "db_real", tf.reduce_mean(self.DB_real))
        self.db_fake_sum = tf.summary.scalar(
            "db_fake", tf.reduce_mean(self.DB_fake))
        self.db_res_sum = tf.summary.scalar(
            "db_res", tf.reduce_mean(self.DB_res))

        self.da_real_sum = tf.summary.scalar(
            "da_real", tf.reduce_mean(self.DA_real))
        self.da_fake_sum = tf.summary.scalar(
            "da_fake", tf.reduce_mean(self.DA_fake))
        self.da_res_sum = tf.summary.scalar(
            "da_res", tf.reduce_mean(self.DA_res))

        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.db_loss_real_sum = tf.summary.scalar(
            "db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar(
            "db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar(
            "da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar(
            "da_loss_fake", self.da_loss_fake)

        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.da_real_sum, self.da_fake_sum, self.da_res_sum,
             self.db_real_sum, self.db_fake_sum, self.db_res_sum,
             self.d_loss_sum]
        )

        # test
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.sig_len, 1], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.sig_len, 1], name='test_B')
        self.testB = self.generator(
            self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(
            self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            print(var.name)

    def train(self):

    def save_model(self, checkpoint_dir, step):
        model_name = "cyclegan_vc.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.sig_len)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_model(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.sig_len)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_test(self):

    def test(self):
