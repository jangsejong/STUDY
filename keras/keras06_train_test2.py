import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# train과 test 비용을 8:2 로 분리하시오.

x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

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
loss :  0.0011513317003846169
[11]의 예측값 :  [[11.040485]]
'''