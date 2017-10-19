import numpy as np

from nn.gan import GAN, ConditionalGAN
from data.dataset import mnist
from data.util import get_noise_sample, get_random_condition

batch_size = 100
sample_size = 100

dataset = mnist()

model = ConditionalGAN()
model.build()

for step in range(1, 100001):
    for _ in range(1):
        noise = get_noise_sample(batch_size, sample_size)
        fake_condition = get_random_condition(batch_size)

        fake_data = model.generator().predict([noise, fake_condition])
        real_data, real_condition = dataset.get_batch(batch_size)

        x_data = np.concatenate((real_data, fake_data))
        x_condition = np.concatenate((real_condition, fake_condition))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = model.discriminator().train_on_batch([x_data, x_condition], y)

    noise = get_noise_sample(batch_size, sample_size)
    condition = get_random_condition(batch_size)
    y = np.ones([batch_size, 1])
    g_loss = model.gan().train_on_batch([noise, condition], y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))
