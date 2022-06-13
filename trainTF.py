import argparse
import os
import tensorflow as tf

import numpy as np
import rawpy

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.activations import softmax

from data_load.data_augment import training_data_generator
from data_load.data_preprocess import normalization, read_image
from unetTF import unet
from unetplus import unet_plus_plus


def decode_and_resize(train_noise_filenames):
    raw_noise = rawpy.imread(train_noise_filenames)
    noise_data = raw_noise.raw_image_visible
    return noise_data

def train(args):
    """
    Training model: unet or unet++.
    """
    train_noise_dir = args.train_noise_dir
    train_gt_dir = args.train_gt_dir
    epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    black_level = args.black_level
    white_level = args.white_level
    patchsize = args.patchsize

    train_noise_filenames = [train_noise_dir + filename for filename in os.listdir(train_noise_dir)]
    train_gt_filenames = [train_gt_dir + filename for filename in os.listdir(train_gt_dir)]


    loss_object = tf.keras.losses.MeanAbsoluteError()
    loss_object1 = tf.keras.losses.KLDivergence()
    lr_all = ExponentialDecay(
        learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
    )
    optimizer_all = tf.keras.optimizers.Adam(lr_all)

    # net = unet((patchsize, patchsize, 4))
    net = unet_plus_plus((patchsize, patchsize, 4))
    net.load_weights('./result/tf_model.h5')

    for i, j in zip(train_noise_filenames, train_gt_filenames):
        print(f"i:{i}, j:{j}")
        noise_c, h, w = read_image(i)
        gt_c, h, w = read_image(j)
        raw_noise_channels_normal = normalization(noise_c, black_level, white_level)
        raw_gt_channels_normal = normalization(gt_c, black_level, white_level)
        del noise_c, gt_c
        raw_noise_channels_normal = tf.convert_to_tensor(np.reshape(raw_noise_channels_normal,
                                                                   (1, raw_noise_channels_normal.shape[0],
                                                                    raw_noise_channels_normal.shape[1], 4)))
        raw_gt_channels_normal = tf.convert_to_tensor(np.reshape(raw_gt_channels_normal,
                                                                   (1, raw_gt_channels_normal.shape[0],
                                                                    raw_gt_channels_normal.shape[1], 4)))
        tr_gen, dtypes, shapes = training_data_generator(
            raw_noise_channels_normal[0], raw_gt_channels_normal[0], patchsize
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        for iii, batch in zip(range(epoch), TRAIN.batch(batch_size)):
            x, y = batch
            with tf.GradientTape() as tape:
                result = net(x, training=True)
                l2_loss = sum(net.losses)
                recon_loss = loss_object(y, result)
                kl_loss = loss_object1(softmax(y), softmax(result))
                total_loss = l2_loss + recon_loss + kl_loss
                targets_all = net.trainable_variables
                gradients_all = tape.gradient(total_loss, targets_all)
                optimizer_all.apply_gradients(zip(gradients_all, targets_all))
        del raw_noise_channels_normal, raw_gt_channels_normal, TRAIN

    net.save('./result/tf_model.h5')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--train_noise_dir', type=str, default="./traindata/noisy/")
    parser.add_argument('--train_gt_dir', type=str, default="./traindata/ground truth/")
    parser.add_argument('--patchsize', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    args = parser.parse_args()
    train(args)

