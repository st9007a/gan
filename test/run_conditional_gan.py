import pickle
import numpy as np

from nn.gan import ConditionalGAN
from data.dataset import mnist
from data.util import get_noise_sample, get_random_condition

batch_size = 100
sample_size = 100

dataset = mnist()

model = ConditionalGAN()
model.build()

for step in range(1, 100001):
    for _ in range(1):
        data = None
        condition = None
        y = None
        if step % 1 == 0:
            noise = get_noise_sample(batch_size, sample_size)
            condition = get_random_condition(batch_size)

            data = model.generator().predict([noise, condition])
            y = np.array([0] * batch_size)
        else:
            data, condition = dataset.get_batch(batch_size)
            y = np.ones([batch_size, 1])

        # x_data = np.concatenate((real_data, fake_data))
        # x_condition = np.concatenate((real_condition, fake_condition))
        # y = np.ones([2 * batch_size, 1])
        # y[batch_size:, :] = 0

        d_loss = model.discriminator().train_on_batch([data, condition], y)

    noise = get_noise_sample(batch_size, sample_size)
    condition = get_random_condition(batch_size)
    y = np.ones([batch_size, 1])
    g_loss = model.gan().train_on_batch([noise, condition], y)

    if step % 100 == 0:
        print("step " + str(step) + " d_loss: " + str(d_loss) + " g_loss: " + str(g_loss))

model.generator().save('./models/cgan-keras-' + str(d_loss) + '-' + str(g_loss))

print('Generate fake data')
noise = get_noise_sample(100, sample_size)
cond = []
for i in range(100):
    cond.append([1 if j == i / 10 else 0 for j in range(10)])
mnist_fake = model.generator().predict([np.array(noise), np.array(cond)])

print(mnist_fake)

with open('./fake_data/cgan-keras-' + str(d_loss) + '-' + str(g_loss) + '.pkl', 'wb') as p:
    pickle.dump(mnist_fake, p, protocol = pickle.HIGHEST_PROTOCOL)
