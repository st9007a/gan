#!/usr/bin/python
import pickle
import sys
import numpy as np
from skimage.io import imsave

raw_file = sys.argv[1]
raw_data = None

with open(raw_file, 'rb') as p:
    raw_data = pickle.load(p)

idx = 0
for data in raw_data:
    imsave('fake_image/' + str(idx) + '.png', np.reshape(data, (28, 28)))
    idx += 1
