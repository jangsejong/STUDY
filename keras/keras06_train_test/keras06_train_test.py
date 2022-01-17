import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

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
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #mne
model.fit(x_train, y_train, epochs=1000, batch_size=1)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 예측값 : ', result)

'''
loss :  1.515824466814808e-12
[11]의 예측값 :  [[10.999999]]
'''