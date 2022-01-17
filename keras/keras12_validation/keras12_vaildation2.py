from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = x[:12]
print(x_train)
y_train = y[:12]
x_test = x[12:14]
y_test = y[12:14]
x_validation = x[15:]
y_validation = y[15:]
'''
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])
x_validation = np.array([14, 15, 16])
y_validation = np.array([14, 15, 16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data = (x_validation, y_validation))

'''
loss : 5.961880447102885e-07
17의 예측값:  [[16.99873]]

'''

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :

y_predict = model.predict([17])
print("17의 예측값: ", y_predict)