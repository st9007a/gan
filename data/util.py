import numpy as np

from random import randint

def get_noise_sample(batch_size, samples):
    return np.random.uniform(-1., 1., size = [batch_size, samples])

def get_random_condition(batch_size):
    cond = []
    for _ in range(batch_size):
        cond.append([1 if randint(0, 9) == i else 0 for i in range(10)])

    return np.array(cond)


