from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import PIL
from matplotlib import pyplot

class IOHandler():
    def __init__(self, path=''):
        self.workdir = path

    def _pathify(self, path):
        p = os.path.join(self.workdir, path)
        if not os.path.exists(p):
            os.makedirs(p)
        return p
        
    def imgs_to_np(self, path):
        path = self._pathify(path)
        all_files = os.listdir(path)
        imgs = np.zeros((len(all_files), 256, 256, 3))
        for i in range(len(all_files)):
            file = all_files[i]
            print(i, end='\r')
            image = np.array(load_img(path+'/'+ file, target_size=(256, 256)))
            image = (image - 127.5) / 127.5
            imgs[i] = image
            del image
            
        return imgs

    def load_npz(self, path):
        path = self._pathify(path)
        # with np.load('real-images.npz', mmap_mode='r+') as data:
        #     photo_imgs = data['photo']
        #     impr_imgs = data['impr']
        photo_imgs = np.load('real-images/photo.npy', mmap_mode='r')
        impr_imgs = np.load('real-images/impr.npy', mmap_mode='r')
        return photo_imgs, impr_imgs

    def save_imgs(self, path, images):
        path = self._pathify(path)
        for i, img in enumerate(images):
            img = PIL.Image.fromarray(np.around(img).astype(np.uint8))
            img.save(path+ '/image-{}.jpg'.format(i))


    def save_gen_models(self, path, step, g_model_AtoB, g_model_BtoA):
        path = self._pathify(path)
        # save the first generator model
        filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
        g_model_AtoB.model.save(+filename1)
        # save the second generator model
        filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
        g_model_BtoA.model.save(path+'/'+filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    def save_models(self, path, step=1, models=[]):
        path = self._pathify(path)
        for model in models:
            f = model.name + '%05d.h5' % (step)
            model.model.save(path+'/'+f)
        
    def save_fig(self, path, X_in, X_out):
        path = self._pathify(path)
        # plot real images
        for i in range(3):
            pyplot.subplot(2, 3, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_in[i])
        # plot translated image
        for i in range(n_samples):
            pyplot.subplot(2, 3, 1 + 3 + i)
            pyplot.axis('off')
            pyplot.imshow(X_out[i])
        # save plot to file
        filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
        pyplot.savefig(path+'/'+filename1)
        pyplot.close()
