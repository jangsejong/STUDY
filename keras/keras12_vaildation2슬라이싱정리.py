from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


x = np.array(range(1,17)) #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
x_train = x[:11]
print(x_train) # [ 1  2  3  4  5  6  7  8  9 10 11]
'''
x = np.array(range(1,17)) #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
x_train = x[1:12]
print(x_train) # [ 2  3  4  5  6  7  8  9 10 11 12]

x = np.array(range(17)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
print(x)
x_train = x[1:12] #[ 1  2  3  4  5  6  7  8  9 10 11]
print(x_train)
'''