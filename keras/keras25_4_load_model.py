# 각각의 Scaler의 특성과 정의 정리해놀것!!!

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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

'''
#2. 모델구성

input1 = Input(shape=(13,))
dense1 = Dense(55)(input1)
dense2 = Dense(45)(dense1)
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(35)(dense3)
dense5 = Dense(30)(dense4)
dense6 = Dense(25, activation='relu')(dense5)
dense7 = Dense(20)(dense6)
dense8 = Dense(15)(dense7)
dense9 = Dense(10, activation='relu')(dense8)
dense10 = Dense(5)(dense9)
dense11 = Dense(2)(dense10)
ouput1 = Dense(1)(dense11)
model = Model(inputs=input1, outputs=ouput1)

model.summary()

#model = load_model("./_save/keras25_3_save_model.h5")



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1100, batch_size=13,
          validation_split=0.1) #validation 사용시 성능이 더 좋아진다.
         # validation_data = (x_val, y_val))

#model.save("./_save/keras25_3_save_model.h5")
'''
model = load_model("./_save/keras25_3_save_model.h5")


#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) 


'''
# MinMaxScaler                       #layer 에 relu 반영시

loss : 16.66873550415039             loss : 8.34216594696045   
r2score : 0.7982411072270883         r2score : 0.8990261525165145   

# StandardScaler
loss : 18.252599716186523            loss : 8.482542037963867       
r2score : 0.7790699606512423         r2score : 0.8973270448642088  

# RobustScaler
loss : 17.646638870239258            loss : 9.703550338745117   
r2score : 0.7864045336056296         r2score : 0.8825479273496296   

# MaxAbsScaler
loss : 16.66651153564453             loss : 9.844630241394043   
r2score : 0.7982680101777346         r2score : 0.880840280687609   

#함수형모델 사용시
# MaxAbsScaler
loss : 9.24857234954834
r2score : 0.8880549780029223

'''
