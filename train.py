import os
import sys
import numpy as np
from models import ModelFactory
from io_handler import IOHandler

class Trainer():
    def __init__(self, image_shape, io):
        self.mf = ModelFactory(image_shape)
        self.io = io

    def train(self, path1, path2):
        print('path to imgs:', path1, " path to arts:", path2)
        g1, g2, d1, d2, c1, c2 = self.mf.training_models()
        io = IOHandler('C:/Users/Jake/Documents/testImpress')
        photo_imgs, impr_imgs = IOHandler().load_npz('real-images.npz')

    def _generate_real_samples(self, dataset, n_samples, patch_shape):
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return X, y

    def _summarize_performance(self, step, g_model, trainX, name, n_samples=5):
        # select a sample of input images
        X_in, _ = self._generate_real_samples(trainX, n_samples, 0)
        # generate translated images
        X_out, _ = g_model.generate_fake_samples(X_in, 0)
        # scale all pixels from [-1,1] to [0,1]
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0
        self.io.save_fig('figs', X_in, X_out)

    def _update_image_pool(self, pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif np.random.random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)

# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    # define properties of the training run
    n_epochs, n_batch, = 25, 1 #* tpu_strategy.num_replicas_in_sync
    # determine the output square shape of the discriminator
    n_patch = d_model_A.model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    print("doing {} steps".format(n_steps))
    print(bat_per_epo)
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = self._generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = self._generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = g_model_BtoA.generate_fake_samples(X_realB, n_patch)
        X_fakeB, y_fakeB = g_model_AtoB.generate_fake_samples(X_realA, n_patch)
        # update fakes from pool
        X_fakeA = self._update_image_pool(poolA, X_fakeA)
        X_fakeB = self._update_image_pool(poolB, X_fakeB)
        
        # update generator B->A via adversarial and cycle loss
        c_model_BtoA.set_trainable(True)
        g_loss2, _, _, _, _  = c_model_BtoA.model.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        c_model_BtoA.set_trainable(False)

        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.model.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.model.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        c_model_AtoB.set_trainable(True)
        g_loss1, _, _, _, _ = c_model_AtoB.model.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        c_model_AtoB.set_trainable(False)

        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.model.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.model.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        s = ('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        sys.stdout.write(s)
        sys.stdout.flush()
        # evaluate the model performance every so often
        if (i+1) % int(300) == 0:
            # plot A->B translation
            self._summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            self._summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (1000) == 0:
            # save the models
            self.io.save_models('models', step=i, models=[d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA])
        

#train(d1, d2, g1, g2, c1, c2, (impr_imgs, photo_imgs))

