from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
datasets = load_diabetes()

#1. 데이터
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=49)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(60, input_dim=10))
model.add(Dense(50))
model.add(Dense(41))
model.add(Dense(33))
model.add(Dense(26))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=20,
          validation_split=0.3) #validation 사용시 성능이 더 좋아진다.
         # validation_data = (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) 
'''
loss : 1996.385498046875
r2score : 0.6253249914600845
'''