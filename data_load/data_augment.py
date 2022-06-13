import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from itertools import count
import numpy as np
import tensorflow as tf

def training_data_generator(x, y, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    c_x, c_y = x.shape[2], y.shape[2]
    chs = c_x + c_y
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    data = tf.concat([x, y], axis=-1)

    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs]

    dtypes = (tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
    )

    return gen, dtypes, shapes
