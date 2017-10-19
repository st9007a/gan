import numpy as np

def get_noise_sample(batch_size, samples):
    return np.random.uniform(-1., 1., size = [batch_size, samples])


