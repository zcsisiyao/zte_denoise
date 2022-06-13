import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np

import dtcwt


def unet_plus_plus(input_shape, out_channel=4, base_filter_num=32):
    relu = 0.2
    inputs = Input(input_shape)
    n, h, w, c = inputs.shape
    h_pad = 32 - h % 32 if not h % 32 == 0 else 0
    w_pad = 32 - w % 32 if not w % 32 == 0 else 0
    padded_image = tf.pad(inputs, [[0, 0], [0, h_pad], [0, w_pad], [0, 0]], "reflect")


    conv0_0 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(padded_image)
    conv0_0 = layers.LeakyReLU(relu)(conv0_0)
    conv0_0 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(conv0_0)
    conv0_0 = layers.LeakyReLU(relu)(conv0_0)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)
    pool1 = dtcwt.tf.Transform2d().forward_channels(conv0_0, "nhwc", nlevels=2, include_scale=False).lowpass

    conv1_0 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv1_0 = layers.LeakyReLU(relu)(conv1_0)
    conv1_0 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_0)
    conv1_0 = layers.LeakyReLU(relu)(conv1_0)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)
    pool2 = dtcwt.tf.Transform2d().forward_channels(conv1_0, "nhwc", nlevels=2, include_scale=False).lowpass

    up1_0 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_0)
    merge00_10 = concatenate([conv0_0,up1_0], axis=-1)
    conv0_1 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(merge00_10)
    conv0_1 = layers.LeakyReLU(relu)(conv0_1)
    conv0_1 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(conv0_1)
    conv0_1 = layers.LeakyReLU(relu)(conv0_1)

    conv2_0 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv2_0 = layers.LeakyReLU(relu)(conv2_0)
    conv2_0 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_0)
    conv2_0 = layers.LeakyReLU(relu)(conv2_0)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)
    pool3 = dtcwt.tf.Transform2d().forward_channels(conv2_0, "nhwc", nlevels=2, include_scale=False).lowpass

    up2_0 = Conv2DTranspose(base_filter_num*2, (2, 2), strides=(2, 2), padding='same')(conv2_0)
    merge10_20 = concatenate([conv1_0,up2_0], axis=-1)
    conv1_1 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge10_20)
    conv1_1 = layers.LeakyReLU(relu)(conv1_1)
    conv1_1 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    conv1_1 = layers.LeakyReLU(relu)(conv1_1)

    up1_1 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_1)
    merge01_11 = concatenate([conv0_0,conv0_1,up1_1], axis=-1)
    conv0_2 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(merge01_11)
    conv0_2 = layers.LeakyReLU(relu)(conv0_2)
    conv0_2 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(conv0_2)
    conv0_2 = layers.LeakyReLU(relu)(conv0_2)

    conv3_0 = Conv2D(base_filter_num*8, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv3_0 = layers.LeakyReLU(relu)(conv3_0)
    conv3_0 = Conv2D(base_filter_num*8, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_0)
    conv3_0 = layers.LeakyReLU(relu)(conv3_0)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)
    pool4 = dtcwt.tf.Transform2d().forward_channels(conv3_0, "nhwc", nlevels=2, include_scale=False).lowpass

    up3_0 = Conv2DTranspose(base_filter_num*4, (2, 2), strides=(2, 2), padding='same')(conv3_0)
    merge20_30 = concatenate([conv2_0,up3_0], axis=-1)
    conv2_1 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge20_30)
    conv2_1 = layers.LeakyReLU(relu)(conv2_1)
    conv2_1 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
    conv2_1 = layers.LeakyReLU(relu)(conv2_1)

    up2_1 = Conv2DTranspose(base_filter_num*2, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    merge11_21 = concatenate([conv1_0,conv1_1,up2_1], axis=-1)
    conv1_2 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge11_21)
    conv1_2 = layers.LeakyReLU(relu)(conv1_2)
    conv1_2 = Conv2D(base_filter_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    conv1_2 = layers.LeakyReLU(relu)(conv1_2)

    up1_2 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_2)
    merge02_12 = concatenate([conv0_0,conv0_1,conv0_2,up1_2], axis=-1)
    conv0_3 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(merge02_12)
    conv0_3 = layers.LeakyReLU(relu)(conv0_3)
    conv0_3 = Conv2D(base_filter_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_3)
    conv0_3 = layers.LeakyReLU(relu)(conv0_3)

    conv4_0 = Conv2D(base_filter_num*16, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv4_0 = layers.LeakyReLU(relu)(conv4_0)
    conv4_0 = Conv2D(base_filter_num*16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_0)
    conv4_0 = layers.LeakyReLU(relu)(conv4_0)

    up4_0 = Conv2DTranspose(base_filter_num*8, (2, 2), strides=(2, 2), padding='same')(conv4_0)
    merge30_40 = concatenate([conv3_0,up4_0], axis = -1)
    conv3_1 = Conv2D(base_filter_num*8, 3, padding = 'same', kernel_initializer = 'he_normal')(merge30_40)
    conv3_1 = layers.LeakyReLU(relu)(conv3_1)
    conv3_1 = Conv2D(base_filter_num*8, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
    conv3_1 = layers.LeakyReLU(relu)(conv3_1)

    up3_1 = Conv2DTranspose(base_filter_num*4, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    merge21_31 = concatenate([conv2_0,conv2_1,up3_1], axis = -1)
    conv2_2 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge21_31)
    conv2_2 = layers.LeakyReLU(relu)(conv2_2)
    conv2_2 = Conv2D(base_filter_num*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_2)
    conv2_2 = layers.LeakyReLU(relu)(conv2_2)

    up2_2 = Conv2DTranspose(base_filter_num*2, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    merge12_22 = concatenate([conv1_0,conv1_1,conv1_2,up2_2], axis = -1)
    conv1_3 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge12_22)
    conv1_3 = layers.LeakyReLU(relu)(conv1_3)
    conv1_3 = Conv2D(base_filter_num*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_3)
    conv1_3 = layers.LeakyReLU(relu)(conv1_3)

    up1_3 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_3)
    merge03_13 = concatenate([conv0_0,conv0_1,conv0_2,conv0_3,up1_3], axis = -1)
    conv0_4 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(merge03_13)
    conv0_4 = layers.LeakyReLU(relu)(conv0_4)
    conv0_4 = Conv2D(base_filter_num, 3, padding = 'same', kernel_initializer = 'he_normal')(conv0_4)
    conv0_4 = layers.LeakyReLU(relu)(conv0_4)

    out = layers.Conv2D(out_channel, (1, 1), padding="same")(conv0_4)
    out_holder = out[:, :h, :w, :]

    net_model = keras.Model(inputs=inputs, outputs=out_holder)
    return net_model

if __name__ == "__main__":
    test_input = tf.convert_to_tensor(np.random.randn(1, 512, 512, 4))
    net = unet_plus_plus((512, 512, 4))
    net.summary()
    output = net(test_input, training=False)
    print("test over")