import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)  #(4, 3) (4,)

#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수

x = x.reshape(4, 3, 1)

#2.모델구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1)))
model.add(Dense(1))