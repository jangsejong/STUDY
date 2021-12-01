from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=49)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, shuffle=True, random_state=66)


#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


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
model.add(Dense(8, activation='relu'))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=5,
          validation_split=0.3) #validation 사용시 성능이 더 좋아진다.
         # validation_data = (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) 
'''
loss : 2046.3428955078125
r2score : 0.6159491268948538

============================
# RobustScaler
loss : 2005.270751953125
r2score : 0.6236574362526468
================
#layer 에 relu 반영시 값이 안좋아진다
loss : 2300.379638671875
r2score : 0.5682723557521253

relu 반영을 줄이니 기존 값과 비슷하게 나왔다
loss : 2036.248779296875
r2score : 0.6178435238910566

batch_size=5 로 변경시 큰 변화가 없었다
loss : 2046.72021484375
r2score : 0.6158783334861431


# MinMaxScaler                       #layer 에 relu 반영시

loss : 2076.62646484375             loss : 2038.6666259765625
r2score : 0.6102656376254028        r2score : 0.6173897994718265 

# StandardScaler
loss : 2037.9686279296875           loss : 3738.35400390625    
r2score : 0.6175207836484327        r2score : 0.2983980678376623 

# RobustScaler
loss : 2005.270751953125            loss : 2300.379638671875 
r2score : 0.6236574362526468        r2score : 0.6178435238910566

# MaxAbsScaler
loss : 2056.781005859375            loss : 2191.87353515625 
r2score : 0.6139901823422093        r2score : 0.5886364211423891


'''
