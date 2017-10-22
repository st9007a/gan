import pickle
import numpy as np

from nn.gan import DCGAN
from data.dataset import mnist
from data.util import get_noise_sample

batch_size = 100
sample_size = 100

dataset = mnist()

model = DCGAN(noise_shape = (sample_size, ))
model.build()

for step in range(1, 10001):
    for _ in range(1):
        noise = get_noise_sample(batch_size, sample_size)

        fake_data = model.generator().predict(noise)
        real_data, _ = dataset.get_batch2D(batch_size)

        data = np.concatenate([real_data, fake_data])
        y = np.ones([batch_size * 2, 1])
        y[batch_size:] = 0

        d_loss = model.discriminator().train_on_batch(data, y)
        print(str(step % 100) + '%', end = '\r')

    noise = get_noise_sample(batch_size, sample_size)
    y = np.ones([batch_size, 1])
    g_loss = model.gan().train_on_batch(noise, y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))

model.generator().save('./models/dcgan-keras-' + str(d_loss) + '-' + str(g_loss))

print('Generate fake data')
noise = get_noise_sample(100, sample_size)
mnist_fake = model.generator().predict(np.array(noise))
np.reshape(mnist_fake, (sample_size, 784))

print(np.array(mnist_fake).shape)

with open('./fake_data/dcgan-keras-' + str(d_loss) + '-' + str(g_loss) + '.pkl', 'wb') as p:
    pickle.dump(mnist_fake, p, protocol = pickle.HIGHEST_PROTOCOL)
