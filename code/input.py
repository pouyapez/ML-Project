import numpy as np
import random

sample_num = 12000
input_length = 100
output = np.random.randint(2, size=(sample_num, input_length))
np.save("out.npy", output)