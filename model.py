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
        # Generator
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

        self.g_loss = tf.reduce_mean(tf.abs(self.DA_fake - tf.ones_like(self.DA_fake)))  \
            + tf.reduce_mean(tf.abs(self.DB_fake - tf.ones_like(self.DB_fake)))
            + self.L1_lambda_tf * \
                tf.reduce_mean(tf.abs(self.real_A - self.fake_A_))
            + self.L1_lambda_tf * \
                tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))

    def train(self):

    def save_model(self):

    def load_model(self):

    def sample_test(self):

    def test(self):
