import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 [input layer]
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성 [hidden layer]
model = Sequential()
model.add(Dense(55, input_dim=1))         # 노드 1개 > 55개
model.add(Dense(77, activation='relu'))          #하이퍼 파라미트 튜닝
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(70, activation='linear'))
model.add(Dense(44))
model.add(Dense(1))

model.summary() # 연산개수확인  , 바이어스로 인해 각 레이어마다 노드 1개가 더 있는것처럼 연산 갯수가 늘어난다.

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #mne
model.fit(x, y, epochs=30, batch_size=1)

#4. 평가, 예측 [output layer]
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


loss :  0.0002002768887905404
4의 예측값 :  [[3.98252]]
'''