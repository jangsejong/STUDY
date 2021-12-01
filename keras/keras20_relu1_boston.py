# 각각의 Scaler의 특성과 정의 정리해놀것!!!

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target
print(np.min(x), np.max(x))  #0.0  711.0   

# x = x/711.             #. 안전하다
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)  #shuffle 은 기본값 True
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(55, input_dim=13))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40, activation='relu'))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25, activation='relu'))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1100, batch_size=13,
          validation_split=0.1) #validation 사용시 성능이 더 좋아진다.
         # validation_data = (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) 

'''
# MinMaxScaler
loss : 16.66873550415039
r2score : 0.7982411072270883
============================
#layer 에 relu 반영시
loss : 8.34216594696045
r2score : 0.8990261525165145
'''
