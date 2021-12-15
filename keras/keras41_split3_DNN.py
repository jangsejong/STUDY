import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, GRU
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
a = np.array(range(1,101))

x_predict = np.array(range(96, 106))

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)- size + 1 ):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)