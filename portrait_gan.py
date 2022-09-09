import numpy as np
import random
import os
import  PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Flatten, ZeroPadding2D, Dropout, Activation, Reshape,BatchNormalization, UpSampling2D, LeakyReLU, Conv2D, Input, Dense, Conv2DTranspose, Concatenate
from keras.optimizers import Adam
import shutil
from keras.utils import array_to_img
from keras import preprocessing

data_path = r"C:\Users\dhabr\OneDrive\Desktop\gans_in_action\archive"
batch_size = 64

data = keras.preprocessing.image_dataset_from_directory(data_path,
 label_mode = None, 
 image_size = (64, 64), 
 batch_size = batch_size).map(lambda x:x/255.0)


latent_dim = 100
g_resolution = 2

generator = Sequential()
generator.add(Dense(4*4*256, activation='relu', input_dim=latent_dim))
generator.add(Reshape((4,4,256)))

generator.add(UpSampling2D())
generator.add(Conv2D(256, kernel_size=3, padding='same'))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation('relu'))

generator.add(UpSampling2D())
generator.add(Conv2D(256, kernel_size=3, padding='same'))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation('relu'))

generator.add(UpSampling2D())
generator.add(Conv2D(256, kernel_size=3, padding='same'))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation('relu'))

generator.add(UpSampling2D())
generator.add(Conv2D(128, kernel_size=3, padding='same'))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation('relu'))
generator.add(Conv2D(3, kernel_size=3, padding='same'))
generator.add(Activation('tanh'))

print(generator.summary())


seed = tf.random.normal([1, latent_dim])
Generated_potrait = generator(seed, training=False)
plt.imshow(Generated_potrait[0,:,:,0])
plt.axis('off')
plt.show()


discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64,64,3), padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
discriminator.add(ZeroPadding2D(padding=((0,1), (0,1))))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation="sigmoid"))

print(discriminator.summary())

Discriminator_Verdit = discriminator(Generated_potrait)
print(Discriminator_Verdit)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        seed = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(seed)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size,1 )), tf.zeros((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        seed = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(seed))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss":self.g_loss_metric.result()}

epochs = 200

discriminator_opt = tf.keras.optimizers.Adamax(1.5*(0.0001), 0.5)
generator_opt = tf.keras.optimizers.Adamax(1.5*(0.0001), 0.5)

loss_fn = keras.losses.BinaryCrossentropy()

model = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
model.compile(d_optimizer=discriminator_opt, g_optimizer=generator_opt, loss_fn=loss_fn)

history = model.fit(data, epochs=epochs)


num_gen_imgs = 50

os.mkdir('generated_images')

def Potrait_Generator():
    Generated_Paintings = []
    seed = tf.random.normal([num_gen_imgs, latent_dim])
    generated_image = generator(seed)
    generated_image = generated_image*255
    generated_image = generated_image.numpy()
    for i in range(num_gen_imgs):
        img = keras.preprocessing.image.array_to_img(generated_image[i])
        Generated_Paintings.append(img)
        img.save("generated_images/Potraits{:02d}.png".format(i))
    return

Images = Potrait_Generator()

shutil.make_archive('generated_images_zip', 'zip', 'generated_images')

