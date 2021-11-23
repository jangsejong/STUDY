import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 [input layer]
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성 [hidden layer]
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))          #하이퍼 파라미트 튜닝
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #mne
model.fit(x, y, epochs=30, batch_size=1)

#4. 평가, 예측 [output layer]
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  3.2657339033903554e-05
4의 예측값 :  [[3.9966502]]..
'''