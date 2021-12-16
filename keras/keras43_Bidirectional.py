import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf

#1. 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

#print(x.shape, y.shape)  #(4, 3) (4,)

#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수

x = x.reshape(4, 3, 1)
#y = y.reshape(4, 1)
#x = np.array(x, dtype=np.float32)
print(y.shape)




#2.모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(32, activation='relu',  return_sequences=True), input_shape=(3, 1)))
# model.add(SimpleRNN(32, activation='relu', input_shape=(3, 1),  return_sequences=True))
# model.add(Bidirectional(SimpleRNN(10, activation='linear')))
model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가,예측
model.evaluate(x, y)
#pre = model.predict(x)
results = model.predict([[[5],[6],[7]]])
#results = model.predict([[5],[6],[7]])
#y_predict=results.round(0).astype(int)
print(results)



