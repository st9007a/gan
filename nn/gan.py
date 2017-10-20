from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, LeakyReLU, PReLU, Conv2D, UpSampling2D, MaxPooling2D, Reshape, Activation, Flatten
from keras.optimizers import Adam

class GAN():

    def __init__(self, noise_shape):
        self.full_model = None
        self.D = None
        self.G = None
        self.model_build = False

        self.noise_shape = noise_shape

    def generator(self):
        if self.G is None:
            self.build()
        return self.G

        return self.G

    def discriminator(self):
        if self.D is None:
            self.build()
        return self.D

    def gan(self):
        if self.full_model is None:
            self.build()
        return self.full_model

    def build(self):
        if self.model_build == True:
            return

        self.G = Sequential()
        self.G.add(Dense(256, input_shape = self.noise_shape, activation = 'relu'))
        self.G.add(Dense(784, activation = 'tanh'))

        self.D = Sequential()
        self.D.add(Dense(256, input_shape = (784, ), activation = 'relu'))
        self.D.add(Dense(1, activation = 'sigmoid'))

        self.full_model = Sequential()
        self.full_model.add(self.G)
        self.full_model.add(self.D)

        self.D.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))
        self.full_model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))

        self.model_build = True

    def save(self, path):
        self.generator().save(path)

class ConditionalGAN(GAN):

    def __init__(self, noise_shape):
        super(ConditionalGAN, self).__init__(noise_shape)

        self._d_kernel = []

    def generator_output(self, noise_input, cond_input):
        in_layer_1 = Dense(200)(noise_input)
        in_layer_1 = PReLU()(in_layer_1)
        in_layer_2 = Dense(1000)(cond_input)
        in_layer_2 = PReLU()(in_layer_2)

        output = concatenate([in_layer_1, in_layer_2])
        output = Dense(512)(output)
        output = LeakyReLU()(output)

        return Dense(784, activation = 'sigmoid')(output)

    def d_kern(self):
        if len(self._d_kernel) > 0:
            return self._d_kernel

        self._d_kernel.append(Dense(240))
        self._d_kernel.append(PReLU())
        self._d_kernel.append(Dense(50))
        self._d_kernel.append(PReLU())
        self._d_kernel.append(PReLU())
        self._d_kernel.append(Dense(1, activation = 'sigmoid'))

        return self._d_kernel

    def discriminator_output(self, data_input, cond_input):
        kern = self.d_kern()

        in_layer_1 = kern[0](data_input)
        in_layer_1 = kern[1](in_layer_1)
        in_layer_2 = kern[2](cond_input)
        in_layer_2 = kern[3](in_layer_2)
        output = concatenate([in_layer_1, in_layer_2])
        for i in range(4, len(kern)):
            output = kern[i](output)

        return output

    def build(self):
        if self.model_build == True:
            return

        noise_input = Input(shape = self.noise_shape)
        data_input = Input(shape = (784, ))
        cond_input = Input(shape = (10, ))

        g_out = self.generator_output(noise_input, cond_input)
        d_out = self.discriminator_output(data_input, cond_input)
        gan_out = self.discriminator_output(g_out, cond_input)

        self.G = Model(inputs = [noise_input, cond_input], outputs = g_out)
        self.D = Model(inputs = [data_input, cond_input], outputs = d_out)
        self.full_model = Model(inputs = [noise_input, cond_input], outputs = gan_out)

        self.D.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))
        self.full_model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))

        self.model_build = True

class DCGAN(GAN):

    def __init__(self, noise_shape):
        super(DCGAN, self).__init__(noise_shape)

    def build(self):
        if self.model_build == True:
            return

        self.G = Sequential()
        self.G.add(Dense(1024, input_shape = self.noise_shape, activation = 'relu'))
        self.G.add(Dense(7 * 7 * 64, activation = 'relu'))
        self.G.add(Reshape(target_shape = (7, 7, 64)))
        self.G.add(UpSampling2D(size = (2, 2)))
        self.G.add(Conv2D(32, (5, 5), padding = 'same'))
        self.G.add(UpSampling2D(size = (2, 2)))
        self.G.add(Activation('relu'))
        self.G.add(Conv2D(1, (5, 5), padding = 'same'))
        self.G.add(Activation('relu'))

        self.D = Sequential()
        self.D.add(Conv2D(32, (5, 5), padding = 'same', input_shape = (28, 28, 1)))
        self.D.add(Activation('relu'))
        self.D.add(MaxPooling2D(pool_size = (2, 2)))
        self.D.add(Conv2D(64, (5, 5)))
        self.D.add(Activation('relu'))
        self.D.add(MaxPooling2D(pool_size = (2, 2)))
        self.D.add(Flatten())
        self.D.add(Dense(1024, activation = 'relu'))
        self.D.add(Dense(1, activation = 'sigmoid'))

        self.full_model = Sequential()
        self.full_model.add(self.G)
        self.full_model.add(self.D)

        self.D.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))
        self.full_model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001))

