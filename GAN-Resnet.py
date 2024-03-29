# Google Colab Sharing Link: https://colab.research.google.com/drive/1Un6iZvniWyQf4jilbRWrYZVpKH3me10r

import tensorflow as tf
import numpy as np
import random


def disc(s,n,r):            #s is the seed; put s = -1 to produce x* and give x* coordinates as a, b in the first part of the function
    if(s==-1):
        a = r
        b = r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n))
        array[mask] += 63
        a = n-r
        b = n-r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = r
        b = n-r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = n-r
        b = r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
    else:
        random.seed(s)
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n))
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
    return array
  

  import numpy as np
import random


def disc_no_overlap(s,n,r):         #produce discs with no overlap to curate the dataset
    random.seed(s)
    a = random.randint(r,n//2)
    b = random.randint(r,n//2)
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array = np.zeros((n, n))
    array[mask] += 63
    a = random.randint(n//2,n-r)
    b = random.randint(n//2,n-r)
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask] += 63
    a = random.randint(n//2+r,n-r)
    b = random.randint(r,n//2-r)
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask] += 63
    a = random.randint(r,n//2-r)
    b = random.randint(n//2+r,n-r)
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask] += 63
    return array

  
def dataset(N, n, r):               #produce dataset, N is number of samples, n is grid size, r is radius
    a = disc(0, n, r)
    a = np.reshape(a, (1, n, n))
    for i in range(1, 8*N//10):
        b = disc(i+1000, n, r)
        b = np.reshape(b, (1, n, n))
        a = np.concatenate((a, b))
    for i in range(8*N//10, N-200):
        b = disc_no_overlap(i+1000, n, r)
        b = np.reshape(b, (1, n, n))
        a = np.concatenate((a, b))
    for i in range(N-200, N):
        b = disc(-1, n, r)
        b = np.reshape(b, (1, n, n))
        a = np.concatenate((a, b))
    return a

        

# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(128,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):      #ResNet

        X_input = Input(shape=(self.latent_dim,))
        X = X_input
        X = Dense(4*4*256, activation='relu')(X)
        X = Reshape((4, 4, 256))(X)
        
        num_conv_layers = 4
        num_res_blocks = 3
        for i in range(num_conv_layers):
            X = UpSampling2D(interpolation="bilinear")(X)
            X = Conv2D(int(X.get_shape().as_list()[-1]/2), kernel_size=3, strides=1, padding="same")(X)
            X = Activation("relu")(X)
            X = BatchNormalization()(X)
            if(num_conv_layers - i <= num_res_blocks):
                ResX = X
                X = BatchNormalization()(X)
                X = Activation("relu")(X)
                print(X.get_shape().as_list())
                X = Conv2D(X.get_shape().as_list()[-1], kernel_size=3, strides=1, padding="same")(X)
                X = BatchNormalization()(X)
                X = Activation("relu")(X)
                X = Conv2D(X.get_shape().as_list()[-1], kernel_size=3, strides=1, padding="same")(X)
                X = Add()([X, ResX])
        X = Conv2D(1, kernel_size=1, strides=1, padding="same")(X)
        X = Activation("tanh")(X)       
            
        model =  Model(inputs = X_input, outputs = X)
        model.summary()
        return model
          
    
    def build_critic(self):     #ResNet

        X_input = Input(shape=self.img_shape)
        X = X_input
        num_conv_layers = 6
        num_res_blocks = 4
        num_channels = [32, 64, 128, 256, 256, 512, 512, 512, 512]
        for i in range(num_conv_layers):
            X = Conv2D(num_channels[i], kernel_size=3, strides=2, padding="same")(X)
            X = LeakyReLU(alpha=0.2)(X)
            if num_conv_layers - i <= num_res_blocks:
                ResX = X
                X = Conv2D(X.get_shape().as_list()[-1], kernel_size=3, strides=1, padding="same")(X)
                X = LeakyReLU(alpha=0.2)(X)
                X = Conv2D(X.get_shape().as_list()[-1], kernel_size=3, strides=1, padding="same")(X)
                X = Add()([X, ResX])
                X = LeakyReLU(alpha=0.2)(X)
        X = Conv2D(int(num_channels[num_conv_layers]/4), kernel_size=1, strides=1, padding="same")(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = Conv2D(1, kernel_size=1, strides=1, padding="same")(X)
     
        model =  Model(inputs = X_input, outputs = X)
        model.summary()
        return model

    def train(self, epochs, batch_size, sample_interval):

        # Load the dataset
        X_train = dataset(40000, 64, 5)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 126.0) / 126.0
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
             

    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.imshow(gen_imgs[0,:,:,0])
        plt.colorbar()
        plt.gray()
        plt.show()
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/wgan_%d.png" % epoch)
        self.generator.save("images/wgangp_model.h5")
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=50000, batch_size=32, sample_interval=400)
    wgan.generator.save("images/wgangp_model.h5")