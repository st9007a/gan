import numpy as np

from nn.gan import GAN
from data.dataset import mnist
from data.util import get_noise_sample

batch_size = 256

dataset = mnist()

model = GAN()
model.build()

for step in range(1, 100001):
    for _ in range(1):
        noise = get_noise_sample(batch_size, 128)

        fake_data = model.generator().predict(noise)
        real_data, _ = dataset.get_batch(batch_size)

        x = np.concatenate((real_data, fake_data))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = model.discriminator().train_on_batch(x, y)

    noise = get_noise_sample(batch_size, 128)
    y = np.ones([batch_size, 1])
    g_loss = model.gan().train_on_batch(noise, y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))
