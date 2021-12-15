import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, GRU
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
a = np.array(range(1,11))
# print(a) #[ 1  2  3  4  5  6  7  8  9 10]
# print(a.shape) #(10,)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)- size +1 ):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)
# print(dataset)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
'''

# bbb = split_x(a, size)

# x = bbb[:,:-1]  
# y = bbb[:, -1] 
# print(x, y) #[6 7 8 9]] [ 5  6  7  8  9 10]
# print(x.shape, y.shape) #(6, 4) (6,)

