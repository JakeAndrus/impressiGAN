import tensorflow as tf
import os
from io_handler import IOHandler


class Generator():
    def __init__(self, io):
        self.io = io
    def generate(self, w, path1, path2):
        print('weights:', w, 'path to img:', path1, " path to folder:", path2)
        impr_gen = tf.keras.models.load_model('models/photo-to-impr-generator-00001.h5')
        #impr_gen = tf.keras.models.load_model('models/g_model_AtoB_000001.h5')
        imgs = self.io.imgs_to_np(path1)
        X_out = impr_gen.predict(imgs)
        X_out = (X_out+1)*127.5
        self.io.save_imgs(path2, X_out)


