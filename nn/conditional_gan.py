from keras.models import Model
from keras.layers import Input, Dense, concatenate, LeakyReLU
from keras.optimizators import Adam

class ConditionalGAN():

    def __init__(self):
        self.G = None
        self.D = None
        self.full_model = None
        self.model_build = False

        self._d_kernel = []

    def generator_output(self, noise_input, cond_input):
        in_layer_1 = Dense(200, activation = 'relu')(noise_input)
        in_layer_2 = Dense(1000, activation = 'relu')(cond_input)

        output = concatenate([in_layer_1, in_layer_2])

        return Dense(784, activation = 'sigmoid')(output)

    def d_kern(self):
        if len(self._d_kernel) > 0:
            return self._d_kernel

        self._d_kernel.append(LeakyReLU(240))
        self._d_kernel.append(LeakyReLU(50))
        self._d_kernel.append(LeakyReLU(240))
        self._d_kernel.append(Dense(1, activation = 'sigmoid'))

        return self._d_kernel

    def discriminator_output(self, data_input, cond_input):
        kern = self.d_kern()

        output = concatenate(kern[0](data_input), kern[1](cond_input))
        for i in range(2, len(kern)):
            output = kern[i](output)

        return output

    def generator(self):
        if self.G is None:
            self.build()
        return self.G

    def discriminator(self):
        if self.D is None:
            self.build()
        return self.D

    def gan(self):
        if self.full_model is None:
            self.build
        return self.full_model

    def build(self):
        if self.model_build == True:
            return

        noise_input = Input(shape = (128, ))
        data_input = Input(shape = (784, ))
        cond_input = Input(shape = (10, ))

        g_out = self.generator_output(noise_input, cond_input)
        d_out = self.discriminator_output(data_input, cond_input)
        gan_out = self.discriminator_output(g_out, cond_input)

        self.G = Model(inputs = [noise_input, cond_input], outputs = g_out)
        self.D = Model(inputs = [data_input, cond_input], outputs = d_out)
        self.full_model = Model(inputs = [noise_input, cond_input], outputs = gan_out)

        self.D.compile(loss = 'binary_crossentropy', optimizator = Adam(0.00015))
        self.full_model.compile(loss = 'binary_crossentropy', optimizator = Adam(0.0001))

        self.model_build = True
