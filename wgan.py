#!/usr/bin/python
import pickle

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 256

def generator():
    model = Sequential()
    model.add(Dense(256, input_dim = 128, activation = 'relu'))
    model.add(Dense(784, activation = 'sigmoid'))

    return model

def discriminator():
    model = Sequential()
    model.add(Dense(256, input_dim = 784, activation = 'relu'))
    model.add(Dense(1))

    return model

def d_loss_func(y_true, y_pred):
    return K.mean(y_pred[:batch_size] - y_pred[batch_size:])

def g_loss_func(y_true, y_pred):
    return -K.mean(y_pred)

d_base_model = discriminator()
g_base_model = generator()

d_model = Sequential()
d_model.add(d_base_model)
d_model.compile(loss = d_loss_func, optimizer = RMSprop(lr = 0.00005, clipvalue = 0.01))

gan_model = Sequential()
gan_model.add(g_base_model)
gan_model.add(d_base_model)
gan_model.compile(loss = g_loss_func, optimizer = RMSprop(0.00005))

for step in range(1, 10001):
    for _ in range(5):
        noise = np.random.uniform(-1., 1., size = [batch_size, 128])
        fake_data = g_base_model.predict(noise)
        real_data = mnist.train.next_batch(batch_size)[0]

        x = np.concatenate((real_data, fake_data))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = d_model.train_on_batch(x, y)

    noise = np.random.uniform(-1., 1., size = [batch_size, 128])
    y = np.ones([batch_size, 1])
    g_loss = gan_model.train_on_batch(noise, y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))

g_base_model.save('./models/wgan-keras-' + str(d_loss) + '-' + str(g_loss))

print('Generate fake data')
noise = np.random.uniform(-1., 1., size = [100, 128])
mnist_fake = g_base_model.predict(noise)

print(mnist_fake)

with open('./fake_data/wgan-keras-' + str(d_loss) + '-' + str(g_loss) + '.pkl', 'wb') as p:
    pickle.dump(mnist_fake, p, protocol = pickle.HIGHEST_PROTOCOL)

