# coding: utf-8

import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import CycleGAN

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir',
                    default='A2B', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int,
                    default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step',
                    type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size',
                    type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size',
                    type=int, default=1e8, help='# images used to train')

parser.add_argument('--frame_length', dest='fl', type=int,
                    default=32768, help='then crop to signal size')
parser.add_argument('--frame_period', dest='fp', type=int,
                    default=4096, help='then move crop point')
parser.add_argument('--conv_dim', dest='conv_dim', type=int, default=64,
                    help='# filters in first conv layer')

parser.add_argument('--lr', dest='lr', type=float,
                    default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float,
                    default=0.5, help='momentum term of adam')

parser.add_argument('--which_direction', dest='which_direction',
                    default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase',
                    default='train', help='train, test')

parser.add_argument('--save_freq', dest='save_freq', type=int,
                    default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,
                    help='print the debug information every print_freq iterations')

parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                    default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir',
                    default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir',
                    default='./test', help='test sample are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float,
                    default=10.0, help='weight on L1 term in objective')

parser.add_argument('--use_L1_freq', dest='use_L1_freq',
                    type=bool, default=False, help='weight on L1 freq change')
parser.add_argument('--L1_freq', dest='L1_freq', type=int,
                    default=10, help='L1 change freq')
parser.add_argument('--L1_another', dest='L1_another',
                    type=float, default=1.0, help='another one weight on L1')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(
        allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = CycleGAN(sess, args)
        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        else:
            model.val(args)


if __name__ == '__main__':
    tf.app.run()
