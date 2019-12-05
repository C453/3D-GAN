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
batch_size = 32
z_size = 200
DIR_PATH = 'train/'
generated_volumes_dir = 'generated_volumes'
log_dir = 'logs'
epochs = 10
NOISE_SHAPE = (1, 1, 1, z_size)
IMAGE_SHAPE = (64, 64, 64, 1)
loss_function = "binary_crossentropy"

# Sourced From
# https://github.com/PacktPublishing/Generative-Adversarial-Networks-Projects/blob/master/Chapter02/run.py
def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)

def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def get_data(obj='airplane', cube_len=64, obj_ratio=1.0):
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
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(filters=256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(filters=128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3DTranspose(filters=64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

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

    generator.compile(loss=loss_function, optimizer=gen_optimizer)
    discriminator.compile(loss=loss_function, optimizer=dis_optimizer)

    discriminator.trainable = False
    noise = Input(shape = NOISE_SHAPE)
    gan = Model(noise, discriminator(generator(noise)))
    gan.compile(loss = loss_function, optimizer = gen_optimizer)

    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))
    
    for epoch in range(epochs):
        print("Epoch:", epoch)

        gen_losses = []
        dis_losses = []
        number_of_batches = int(data.shape[0] / batch_size)
        print("Number of batches:", number_of_batches)
        for index in range(number_of_batches):
            z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            batch = data[index * batch_size:(index + 1) * batch_size, :, :, :]
            
            gen_volumes = generator.predict(z_sample)
            
            discriminator.trainable = True

            if index % 2 == 0:
                loss_real = discriminator.train_on_batch(batch, labels_real)
                loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

                d_loss = 0.5 * np.add(loss_real, loss_fake)
            else:
                d_loss = 0.0
            
            discriminator.trainable = False
            
            z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            g_loss = gan.train_on_batch(z, labels_real)

            gen_losses.append(g_loss)
            dis_losses.append(d_loss)
            print("Epoch " + str(epoch) + ", Batch: " + str(index + 1) + ", d_loss: " + str(d_loss) + ", g_loss: " + str(g_loss))

    generator.save_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"))
    discriminator.save_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"))

def main():
    print("Getting Data")
    data = get_data()
    print("Building GAN")
    gan = build_GAN(data)

if __name__ == '__main__':
    main()
