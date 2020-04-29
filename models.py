import tensorflow as tf
import os
import PIL
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow_addons as tfa
from matplotlib import pyplot

class BaseNeuralNetwork:
    def __init__(self, name, input_shape):
        self.name = name
        self.model = None
        self.random_normal = RandomNormal(stddev=.02)
        self.input_layer = Input(shape=input_shape)
        self.output_layer = None

    def model_summary(self):
        if self.model:
            print(self.name)
            print(self.model.summary())
        else:
            print("{} is not built yet".format(self.name))
            
    #create_conv_block is a bit of a template function used by the generator and discriminator models,
    #which need to add chunks of layers with slightly different attributes at different times
    def create_conv_block(self, input_layer, output_space, kernal_size, instance_norm=True, relu='leaky', strides=(2, 2), transpose=False):
        if transpose:
            conv = Conv2DTranspose(output_space, 
                                   kernal_size, 
                                   strides=strides, 
                                   padding='same', 
                                   kernel_initializer=self.random_normal)(input_layer)
        else:
            conv = Conv2D(output_space, 
                          kernal_size, 
                          strides=strides, 
                          padding='same', 
                          kernel_initializer=self.random_normal)(input_layer)
            
        if instance_norm: conv = tfa.layers.InstanceNormalization(axis=-1)(conv)
        #if instance_norm: conv = InstanceNormalization(axis=-1)(conv) 
        if relu == 'leaky': conv = LeakyReLU(alpha=.2)(conv)
        if relu == 'normal': conv = Activation('relu')(conv)
        return conv
        
        
class Discriminator(BaseNeuralNetwork):
    def __init__(self, name, input_shape):
        super().__init__(name, input_shape)
        
    def paper_build(self):    
        #layers are added in a decorator fashion, where the previous layers are passed in to the creation of the new layers
        c = self.create_conv_block(self.input_layer, 64, (4, 4), instance_norm=False)
        c = self.create_conv_block(c, 128, (4, 4))
        c = self.create_conv_block(c, 256, (4, 4))
        #c = self.create_conv_block(c, 256, (4, 4), strides=(1, 1))
        
        c = self.create_conv_block(c, 512, (4, 4))
        c = self.create_conv_block(c, 512, (4, 4), strides=(1, 1))
        out = self.create_conv_block(c, 1, (4, 4), instance_norm=False, relu=None, strides=(1, 1))
        self.output_layer = out
    
    def compile_model(self):
        if self.output_layer is not None:
            self.model = Model(self.input_layer, self.output_layer)
            self.model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        else:
            print('No output layers provided for {}'.format(self.name))
            
class Generator(BaseNeuralNetwork):
    def __init__(self, name, input_shape):
        super().__init__(name, input_shape)
        
    def create_resnet_block(self, input_layer, output_space):
        #first layer
        res = self.create_conv_block(input_layer, output_space, (3, 3), relu='normal', strides=(1, 1))
        # second convolutional layer
        res = self.create_conv_block(res, output_space, (3, 3), relu=None, strides=(1, 1))
        # concatenate merge channel-wise with input layer
        res = Concatenate()([res, input_layer])
        return res
    
    def paper_build(self):
        c = self.create_conv_block(self.input_layer, 64, (7, 7), relu='normal', strides=(1, 1))
        c = self.create_conv_block(c, 128, (3, 3), relu='normal')
        c = self.create_conv_block(c, 256, (3, 3), relu='normal')
        for _ in range(6):
            c = self.create_resnet_block(c, 256)
        c = self.create_conv_block(c, 128, (3, 3), relu='normal', transpose=True)
        c = self.create_conv_block(c, 64, (3, 3), relu='normal', transpose=True)
        c = self.create_conv_block(c, 3, (7, 7), relu=None, strides=(1, 1))
        out = Activation('tanh')(c)
        self.output_layer = out
    
    def compile_model(self):
        if self.output_layer is not None:
            self.model = Model(self.input_layer, self.output_layer)
        else:
            print('No output layers provided for {}'.format(self.name))
            
    def generate_fake_samples(self, dataset, patch_shape):
        X = self.model.predict(dataset)
        #fake images have label of zero
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

class Composite():
    def __init__(self, name, input_shape, g1, d1, g2):
        self.name = name
        self.g_model_1 = g1.model
        self.d_model = d1.model
        self.g_model_2 = g2.model
        
        #discriminator element
        #generator 1 creates an image for domain 1 and discriminator 1 try to see if it was real or fake
        input_gen = Input(shape=input_shape)
        gen1_out = self.g_model_1(input_gen)
        output_d = self.d_model(gen1_out)
        
        #identity element
        #generator 1 receives a real image from domain 1 and trys not to change it
        input_id = Input(shape=input_shape)
        output_id = self.g_model_1(input_id)
        
        #forward cycle
        #generator 2 receives an image from generator 1 and tries to change it back to domain 2
        output_f = self.g_model_2(gen1_out)
        
        #backward cycle
        #generator 2 receives real image from domain 1 and gives it to generator 1 who tries to change it back to domain 1
        gen2_out = self.g_model_2(input_id)
        output_b = self.g_model_1(gen2_out)
        
        self.model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    def set_trainable(self, train):
        if train:
            #We are only training the first generator
            self.g_model_1.trainable = True
            self.d_model.trainable = False
            self.g_model_2.trainable = False
        else:
            #We are only training the first generator
            self.g_model_1.trainable = False
            self.d_model.trainable = True
            self.g_model_2.trainable = True
            
    def compile_model(self):
        #1, 5, 10, 10
        self.model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[5, 3, 7, 7], optimizer=Adam(lr=.0002, beta_1=.5))
        
    def model_summary(self):
        if self.model:
            print(self.name)
            print(self.model.summary())
        else:
            print("{} is not built yet".format(self.name))

#Factory pattern for returning models
class ModelFactory():
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def build_model(self, model_type, model_name='image', g1=None, d1=None, g2=None):
        m = None
        if model_type == 'generator':
            m = Generator('{}-generator'.format(model_name), self.image_shape)
            m.paper_build()
        elif model_type == 'discriminator':
            m = Discriminator('{}-discriminator'.format(model_name), self.image_shape)
            m.paper_build()
        elif model_type == 'composite':
            m = Composite('{}-composite'.format(model_name), self.image_shape, g1, d1, g2)
        else:
            print('invalid model type given: {}'.format(model_type))
       
        try:
            m.compile_model()
        except:
            print('unable to compile model: {}'.format(model_name))

        return m

    def training_models(self, A='impr', B='photo'):
        AtoB = '{}-to-{}'.format(A, B)
        BtoA = '{}-to-{}'.format(B, A)
        g1 = self.build_model('generator', model_name=AtoB)
        g2 = self.build_model('generator', model_name=BtoA)
        d1 = self.build_model('discriminator', model_name=AtoB)
        d2 = self.build_model('discriminator', model_name=BtoA)
        c1 = self.build_model('composite', model_name=AtoB, g1=g1, d1=d1, g2=g2)
        c2 = self.build_model('composite', model_name=BtoA, g1=g2, d1=d2, g2=g1)
        return g1, g2, d1, d2, c1, c2

    def generate_models(self, A='impr', B='photo'):
        AtoB = '{}-to-{}'.format(A, B)
        BtoA = '{}-to-{}'.format(B, A)
        g1 = self.build_model('generator', model_name=AtoB)
        g2 = self.build_model('generator', model_name=BtoA)
        return g1, g2