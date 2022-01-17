import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)  #(13, 3) (13,)

#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수

x = x.reshape(13, 3, 1)

#x = np.array(x, dtype=np.float32)
print(x.shape)



model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(3, 1)))
model.add(Dense(16, activation='linear'))
# model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
# model.add(Dense(2, activation='linear'))
model.add(Dense(1))


#3. 컴파일, 훈련



model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)



#4. 평가,예측
model.evaluate(x, y)
#pre = model.predict(x)
results = model.predict([[[50],[60],[70]]])

#results=results.round(0).astype(int)
print(results)
'''
#[[80]]
[[80.37639]]
'''