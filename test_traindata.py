import os
import numpy as np
import rawpy
import tensorflow as tf
import skimage.metrics
from matplotlib import pyplot as plt
from unetTF import unet
import argparse
from data_load.data_postprocess import inv_normalization, write_image, write_back_dng
from data_load.data_preprocess import normalization, read_image
from unetplus import unet_plus_plus


def denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level):
    """
    Example: obtain ground truth
    """
    gt = rawpy.imread(ground_path).raw_image_visible

    """
    pre-process
    """
    raw_data_channels, height, width = read_image(input_path)
    raw_data_channels_normal = normalization(raw_data_channels, black_level, white_level)
    raw_data_channels_normal = tf.convert_to_tensor(np.reshape(raw_data_channels_normal,
                                                               (1, raw_data_channels_normal.shape[0],
                                                                raw_data_channels_normal.shape[1], 4)))
    # net = unet((raw_data_channels_normal.shape[1], raw_data_channels_normal.shape[2], 4))
    net = unet_plus_plus((raw_data_channels_normal.shape[1], raw_data_channels_normal.shape[2], 4))
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

    """
    obtain psnr and ssim
    """
    psnr = skimage.metrics.peak_signal_noise_ratio(
        gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
    ssim = skimage.metrics.structural_similarity(
        gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)
    print('psnr:', psnr)
    print('ssim:', ssim)

    """
    Shows your input or gt or result image
    """
    f0 = rawpy.imread(ground_path)
    f1 = rawpy.imread(input_path)
    f2 = rawpy.imread(output_path)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(f0.postprocess(use_camera_wb=True))
    axarr[1].imshow(f1.postprocess(use_camera_wb=True))
    axarr[2].imshow(f2.postprocess(use_camera_wb=True))
    axarr[0].set_title('gt')
    axarr[1].set_title('noisy')
    axarr[2].set_title('de-noise')
    plt.show()


def main(args):
    """
    The PSNR and SSIM of the training samples were obtained.
    """
    model_path = args.model_path
    black_level = args.black_level
    white_level = args.white_level
    input_path = args.input_path
    output_path = args.output_path
    ground_path = args.ground_path
    denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./result/tf_model.h5")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--input_path', type=str, default="./traindata/noise.dng ")
    parser.add_argument('--output_path', type=str, default="./traindata/result/denoise.dng")
    parser.add_argument('--ground_path', type=str, default="./traindata/gt.dng")

    args = parser.parse_args()
    main(args)
