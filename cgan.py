#!/usr/bin/python
import pickle
import sys

import tensorflow as tf
import numpy as np

from random import randint
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 256

def get_noise():
    noise = np.random.uniform(-1., 1., size = [batch_size, 128])
    cond = []
    for _ in range(batch_size):
        cond.append([1 if randint(0, 9) == i else 0 for i in range(10)])

    return np.array(noise), np.array(cond)

noise_in = Input(shape = (128, ))
g_cond_in = Input(shape = (10, ))

g_in = concatenate([noise_in, g_cond_in])
g_out = Dense(256, activation = 'relu')(g_in)
g_out = Dense(512, activation = 'relu')(g_out)
g_out = Dense(784, activation = 'sigmoid')(g_out)

data_in = Input(shape = (784, ))
d_cond_in = Input(shape = (10, ))

d_in = concatenate([data_in, d_cond_in])

d_layer_1 = Dense(512, activation = 'relu')
d_layer_2 = Dense(256, activation = 'relu')
d_layer_3 = Dense(1, activation = 'sigmoid')

d_out = d_layer_1(d_in)
d_out = d_layer_2(d_out)
d_out = d_layer_3(d_out)

gan_in = concatenate([g_out, g_cond_in])
gan_out = d_layer_1(gan_in)
gan_out = d_layer_2(gan_out)
gan_out = d_layer_3(gan_out)

g_model = Model(inputs = [noise_in, g_cond_in], outputs = g_out)
d_model = Model(inputs = [data_in, d_cond_in], outputs = d_out)
gan_model = Model(inputs = [noise_in, g_cond_in], outputs = gan_out)

d_model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0001))
gan_model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))

for step in range(1, 100001):
    for _ in range(5):
        noise, cond = get_noise()

        fake_data = g_model.predict([noise, cond])
        real_raw_data = mnist.train.next_batch(batch_size)

        x_data = np.concatenate((real_raw_data[0], fake_data))
        x_cond = np.concatenate((real_raw_data[1], cond))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = d_model.train_on_batch([x_data, x_cond], y)

    noise, cond = get_noise()
    y = np.ones([batch_size, 1])
    g_loss = gan_model.train_on_batch([noise, cond], y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))

g_model.save('./models/cgan-keras-' + str(d_loss) + '-' + str(g_loss))

print('Generate fake data')
noise = np.random.uniform(-1., 1., size = [100, 128])
cond = []
for i in range(100):
    cond.append([1 if j == i / 10 else 0 for j in range(10)])
mnist_fake = g_model.predict([np.array(noise), np.array(cond)])

print(mnist_fake)

with open('./fake_data/cgan-keras-' + str(d_loss) + '-' + str(g_loss) + '.pkl', 'wb') as p:
    pickle.dump(mnist_fake, p, protocol = pickle.HIGHEST_PROTOCOL)

