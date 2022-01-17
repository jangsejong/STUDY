import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]) #(10, 1)
x = np.transpose(x)
print(x.shape) #(1, 10)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) #(3, 10)
y = np.transpose(y)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(80))          #하이퍼 파라미트 튜닝
model.add(Dense(100))
model.add(Dense(250))
model.add(Dense(190))
model.add(Dense(160))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(3))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') #mne
model.fit(x, y, epochs=3000, batch_size=1)


#4. 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([[9]])
print('[9]의 예측값 : ', y_predict)

'''
loss :  0.006239230744540691
[9]의 예측값 :  [[9.978157  1.4931258 0.9776811]]
'''