import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
from loguru import logger

class Autoencoder(Model):
  def __init__(self, latent_dim, input_shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(12000, activation='sigmoid'),
      layers.Reshape(input_shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
    # decoder.add(layers.UpSampling1D(2))
    # logger.debug(decoder.summary())

def autoencoder(x_train):
    latent_dim = 64
    conv_autoencoder = Autoencoder(latent_dim, (x_train.shape[1], x_train.shape[2]))
    
    conv_autoencoder.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['acc'])

    history = conv_autoencoder.fit(x_train, x_train, epochs=100, validation_split=0.2)


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('train.png')

    encoded_imgs = conv_autoencoder.encoder(x_train).numpy()
    decoded_imgs = conv_autoencoder.decoder(encoded_imgs).numpy()
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        #plt.imshow(x_train[i])
        plt.plot(range(len(x_train[i,:,1])), tf.squeeze(x_train[i,:,1]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        #plt.imshow(decoded_imgs[i])
        plt.plot(range(len(x_train[i,:,1])), tf.squeeze(decoded_imgs[i,:,1]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.savefig('recon.png')

    return conv_autoencoder