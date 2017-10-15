#!/usr/bin/python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 256
g_dim = 128

x_d = tf.placeholder(tf.float32, shape = [None, 784])
x_g = tf.placeholder(tf.float32, shape = [None, 128])

weights = {
    'w_d1': weight_variable([784, 128]),
    'w_d2': weight_variable([128, 1]),
    'w_g1': weight_variable([128, 256]),
    'w_g2': weight_variable([256, 784]),
}

bias = {
    'b_d1': bias_variable([128]),
    'b_d2': bias_variable([1]),
    'b_g1': bias_variable([256]),
    'b_g2': bias_variable([784]),
}

var_d = [weights['w_d1'], weights['w_d2'], bias['b_d1'], bias['b_d2']]
var_g = [weights['w_g1'], weights['w_g2'], bias['b_g1'], bias['b_g2']]

def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights['w_g1']), bias['b_g1']))
    h_g2 = tf.nn.sigmoid(tf.add(tf.matmul(h_g1, weights['w_g2']), bias['b_g2']))
    return h_g2

def discriminator(x):
    h_d1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w_d1']), bias['b_d1']))
    h_d2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d1, weights['w_d2']), bias['b_d2']))
    return h_d2

def sample_z(m, n):
    return np.random.uniform(-1., 1., size = [m, n])

g_sample = generator(x_g)
d_real = discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. -d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

d_opt = tf.train.AdamOptimizer(0.0005).minimize(d_loss, var_list = var_d)
g_opt = tf.train.AdamOptimizer(0.0005).minimize(g_loss, var_list = var_g)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for step in range(1, 5001):
    for k in range(5):
        batch_x = mnist.train.next_batch(batch_size)[0]
        _, d_loss_train = sess.run([d_opt, d_loss], feed_dict = {x_d: batch_x, x_g: sample_z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_opt, g_loss], feed_dict = {x_g: sample_z(batch_size, g_dim)})

    if step % 100 == 0:
        print('step ' + str(step) + ' d_loss: ' + str(d_loss_train) + ' g_loss: ' + str(g_loss_train))
