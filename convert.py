import tensorflow as tf
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Input
import tensorflow_addons as tfa
from matplotlib import pyplot
from io_handler import IOHandler

io = IOHandler('C:/Users/Jake/Documents/testImpress')

impr_gen = tf.keras.models.load_model('weights/g_model_AtoB_000001.h5')

test_imgs = io.imgs_to_np('test-pics')
X_out = impr_gen.predict(test_imgs)
X_out = (X_out+1)*127.5

io.save_imgs('gen-pics', X_out)