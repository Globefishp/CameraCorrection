import hello
import numpy as np

a = np.ones((10,10))
b = np.zeros((10,))

hello.process_data_numba(a, b)
print(b)