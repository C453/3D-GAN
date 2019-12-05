import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv3D, UpSampling3D, Activation, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import scipy.io as io
import scipy.ndimage as nd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

import time
import os
import random

tf.compat.v1.disable_eager_execution()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

gen_learning_rate = 0.0025
dis_learning_rate = 10e-5
beta = 0.5
z_size = 200
DIR_PATH = 'train/'
generated_volumes_dir = 'generated_volumes'
log_dir = 'logs'
NOISE_SHAPE = (1, 1, 1, z_size)
IMAGE_SHAPE = (64, 64, 64, 1)
loss_function = "binary_crossentropy"
sample_interval = 100

# Sourced From
# https://github.com/PacktPublishing/Generative-Adversarial-Networks-Projects/blob/master/Chapter02/run.py
def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='cornflowerblue', marker='s')
    plt.savefig(path)
    plt.close()

def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def get_data(obj='airplane', cube_len=64, obj_ratio=0.7):
    obj_path = DIR_PATH + obj + '/30/'
    obj_path += 'train/'
    fileList = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    data = np.asarray([getVoxelsFromMat(obj_path + f, cube_len) for f in fileList], dtype=np.bool)
    data = data[..., np.newaxis].astype(np.float)
    return data

def build_generator(name="generator"):
    model = Sequential()
        
    model.add(Conv3DTranspose(filters=512, kernel_size=4, strides=1, input_shape=NOISE_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.summary()

    noise = Input(shape=(NOISE_SHAPE))
    img = model(noise)
    return Model(noise, img)

def build_discriminator(name="discriminator"):
    model = Sequential()
    
    model.add(Conv3D(filters=64, kernel_size=4, strides=2, padding='same', input_shape=IMAGE_SHAPE))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(filters=128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(filters=256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(filters=512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(filters=1, kernel_size=4, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.summary()

    inputTensor = Input(shape = IMAGE_SHAPE)
    return Model(inputTensor, model(inputTensor))

def build_GAN(data):
    generator = build_generator()
    discriminator = build_discriminator()

    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)

    generator.compile(loss=loss_function, optimizer=gen_optimizer, metrics=['accuracy'])
    discriminator.compile(loss=loss_function, optimizer=dis_optimizer, metrics=['accuracy'])

    discriminator.trainable = False
    noise = Input(shape = NOISE_SHAPE)
    gan = Model(noise, [generator(noise), discriminator(generator(noise))])
    gan.compile(loss='binary_crossentropy', loss_weights=[0.999, 0.001], optimizer = gen_optimizer)
    return generator, discriminator, gan

def train(data, generator, discriminator, gan, epochs=40000, batch_size=32):
    for epoch in range(epochs):
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        number_of_batches = int(data.shape[0] / batch_size)
        print("Number of batches:", number_of_batches)
        for batch_idx in range(number_of_batches):
            z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size, :, :, :]
            
            gen_volumes = generator.predict(z_sample)
            
            # discriminator.trainable = True

            #if batch_idx % 2 == 0:
            loss_real = discriminator.train_on_batch(batch, labels_real)
            loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

            d_loss = 0.5 * np.add(loss_real, loss_fake)
            #else:
            #    d_loss = 0.0
            
            # discriminator.trainable = False
            
            z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            g_loss = gan.train_on_batch(z, [batch, labels_real])

            print("Epoch " + str(epoch) + ", Batch: " + str(batch_idx + 1) + ", d_loss: " + str(d_loss) + ", g_loss: " + str(g_loss))
            if batch_idx % sample_interval == 0:
                z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                generated_volumes = generator.predict(z_sample2, verbose=3)
                for i, generated_volume in enumerate(generated_volumes[:5]):
                    voxels = np.squeeze(generated_volume)
                    voxels[voxels < 0.5] = 0.
                    voxels[voxels >= 0.5] = 1.
                    saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, batch_idx, i))

    generator.save_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"))
    discriminator.save_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"))

def main():
    print("Getting Data")
    data = get_data(obj='chair')
    print("Building GAN")
    generator, discriminator, gan = build_GAN(data)
    print("Training GAN")
    train(data, generator, discriminator, gan, epochs=40000)

if __name__ == '__main__':
    main()
