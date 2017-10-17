#!/usr/bin/python
import pickle
import sys

import tensorflow as tf
import numpy as np

from random import randint
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, concatenate, maximum
from keras.optimizers import SGD

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
noise_size = 100

def get_noise():
    noise = np.random.uniform(-1., 1., size = [batch_size, noise_size])
    cond = []
    for _ in range(batch_size):
        cond.append([1 if randint(0, 9) == i else 0 for i in range(10)])

    return np.array(noise), np.array(cond)

noise_in = Input(shape = (noise_size, ))
g_cond_in = Input(shape = (10, ))

g_noise_in_layer_1 = Dense(200, activation = 'relu')(noise_in)
g_cond_in_layer_1 = Dense(1000, activation = 'relu')(g_cond_in)
g_in = concatenate([g_noise_in_layer_1, g_cond_in_layer_1])
# g_out = Dropout(0.5)(g_in)
g_out = Dense(784, activation = 'sigmoid')(g_in)

data_in = Input(shape = (784, ))
d_cond_in = Input(shape = (10, ))

d_data_in_maxout = [Dense(240) for _ in range(5)]
d_cond_in_maxout = [Dense(50) for _ in range(5)]
d_layer_maxout = [Dense(240) for _ in range(5)]

d_dropout = Dropout(0.5)
d_output_layer = Dense(1, activation = 'sigmoid')

d_in_1 = maximum([layer(data_in) for layer in d_data_in_maxout])
d_in_2 = maximum([layer(d_cond_in) for layer in d_cond_in_maxout])
d_in = concatenate([d_in_1, d_in_2])
d_out = maximum([layer(d_in) for layer in d_layer_maxout])
# d_out = d_dropout(d_out)
d_out = d_output_layer(d_in)

gan_in_1 = maximum([layer(g_out) for layer in d_data_in_maxout])
gan_in_2 = maximum([layer(g_cond_in) for layer in d_cond_in_maxout])
gan_in = concatenate([gan_in_1, gan_in_2])
gan_out = maximum([layer(gan_in) for layer in d_layer_maxout])
# gan_out = d_dropout(gan_out)
gan_out = d_output_layer(gan_in)

g_model = Model(inputs = [noise_in, g_cond_in], outputs = g_out)
d_model = Model(inputs = [data_in, d_cond_in], outputs = d_out)
gan_model = Model(inputs = [noise_in, g_cond_in], outputs = gan_out)

d_opt = SGD(lr = 0.0005)
gan_opt = SGD(lr = 0.0005)

d_model.compile(loss = 'binary_crossentropy', optimizer = d_opt)
gan_model.compile(loss = 'binary_crossentropy', optimizer = gan_opt)

for step in range(1, 100001):
    for _ in range(1):
        noise, cond = get_noise()

        fake_data = g_model.predict([noise, cond])
        real_raw_data = mnist.train.next_batch(batch_size)

        x_data = np.concatenate((real_raw_data[0], fake_data))
        x_cond = np.concatenate((real_raw_data[1], cond))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = d_model.train_on_batch([x_data, x_cond], y)
        # d_opt.momentum *= 5.5e-6
        # d_opt.lr /= 1.00004

    noise, cond = get_noise()
    y = np.ones([batch_size, 1])
    g_loss = gan_model.train_on_batch([noise, cond], y)

    # gan_opt.momentum *= 5.5e-6
    # gan_opt.lr /= 1.00004

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))

g_model.save('./models/cgan-keras-' + str(d_loss) + '-' + str(g_loss))

print('Generate fake data')
noise = np.random.uniform(-1., 1., size = [100, noise_size])
cond = []
for i in range(100):
    cond.append([1 if j == i / 10 else 0 for j in range(10)])
mnist_fake = g_model.predict([np.array(noise), np.array(cond)])

print(mnist_fake)

with open('./fake_data/cgan-keras-' + str(d_loss) + '-' + str(g_loss) + '.pkl', 'wb') as p:
    pickle.dump(mnist_fake, p, protocol = pickle.HIGHEST_PROTOCOL)

