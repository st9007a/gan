import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

class GAN():

    def __init__(self):
        self.full_model = None
        self.D = None
        self.G = None
        self.model_build = False

    def generator(self):
        if self.G is not None:
            return self.G

        self.G = Sequential()
        self.G.add(Dense(256, input_dim = 128, activation = 'relu'))
        self.G.add(Dense(784, activation = 'sigmoid'))

        return self.G

    def discriminator(self):
        if self.D is not None:
            return self.D

        self.D = Sequential()
        self.D.add(Dense(256, input_dim = 784, activation = 'relu'))
        self.D.add(Dense(1, activation = 'sigmoid'))

        return self.D

    def gan(self):
        if self.full_model is None:
            self.build()
        return self.full_model

    def build(self):
        if self.model_build == True:
            return

        self.full_model = Sequential()
        self.full_model.add(self.generator())
        self.full_model.add(self.discriminator())

        self.D.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))
        self.full_model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))

        self.model_build = True

    def save(self, path):
        self.generator().save_model(path)
