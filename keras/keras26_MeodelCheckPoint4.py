from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import time as time
from tensorflow.python.keras.callbacks import History

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
#print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:05d}={val_loss:.5f}.hdf5'
model_path = "".join([filepath, 'k26', datetime,'_', filename])

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(np.min(x), np.max(x))  #0.0  711.0   

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)  #shuffle 은 기본값 True

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

# model = Sequential()
# model.add(Dense(55, input_dim=13))
# model.add(Dense(50))
# model.add(Dense(45))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(35))
# model.add(Dense(30))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(15))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5))
# model.add(Dense(2))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # EarlyStopping patience(기다리는 횟수)

# es = EarlyStopping(monitor='val_loss', patience=10, mode = 'min', verbose = 1) # restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size = 13, 
#                  validation_split = 0.2 , callbacks = [es, mcp])
# end = time.time()- start

#print("걸린시간 : ", round(end, 3), '초')

#model.save_weights("./_save/keras25_1_save_weights.h5")
#model = load_model('./_save/keras26_1_save_MCP.h5')
#model=load_model("./_save/keras26_1_save_MCP.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
loss:  15.390331268310547
r2스코어 :  0.813714939604898

loss:  15.390331268310547
r2스코어 :  0.813714939604898

loss:  15.390331268310547
r2스코어 :  0.813714939604898
'''

model2=load_model("./_save/keras26_3_save_MCP.h5")

