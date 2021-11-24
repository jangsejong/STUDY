import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
print(x.shape)
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(y.shape)


'''
#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(4))          #하이퍼 파라미트 튜닝
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') #mne
model.fit(x, y, epochs=30, batch_size=1)



#4. 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3]])
print('[10, 1.3]의 예측값 : ', y_predict)
'''