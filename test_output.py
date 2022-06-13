import os
import numpy as np
import tensorflow as tf

from data_load.data_postprocess import inv_normalization, write_image, write_back_dng
from data_load.data_preprocess import normalization, read_image
from unetTF import unet
import argparse

from unetplus import unet_plus_plus



def denoise_raw(input_path, output_path, model_path, black_level, white_level):

    """
    pre-process
    """
    raw_data_channels, height, width = read_image(input_path)
    raw_data_channels_normal = normalization(raw_data_channels, black_level, white_level)
    raw_data_channels_normal = tf.convert_to_tensor(np.reshape(raw_data_channels_normal,
                                                               (1, raw_data_channels_normal.shape[0],
                                                                raw_data_channels_normal.shape[1], 4)))
    net = unet_plus_plus((raw_data_channels_normal.shape[1], raw_data_channels_normal.shape[2], 4))
    # net = unet((raw_data_channels_normal.shape[1], raw_data_channels_normal.shape[2], 4))

    if model_path is not None:
        net.load_weights(model_path)

    """
    inference
    """
    result_data = net(raw_data_channels_normal, training=False)
    result_data = result_data.numpy()

    """
    post-process
    """
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    write_back_dng(input_path, output_path, result_write_data)


def main(args):
    """
    Output test file.
    """
    model_path = args.model_path
    black_level = args.black_level
    white_level = args.white_level
    train_noise_dir = './data/noise/'
    train_gt_dir = './data/result/data/'
    train_noise_filenames = [train_noise_dir + filename for filename in os.listdir(train_noise_dir)]
    train_gt_filenames = [train_gt_dir + filename for filename in os.listdir(train_gt_dir)]
    for i,j in zip(train_noise_filenames, train_gt_filenames):
        denoise_raw(i, j, model_path, black_level, white_level)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./result/tf_model.h5")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)

    args = parser.parse_args()
    main(args)
