import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Flatten, Conv1D
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

# print(x.shape, y.shape)  #(581012, 54) (581012,)
x = x.reshape(581012, 9, 6)

import pandas as pd
y = pd.get_dummies(y)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(30,2, activation='linear', input_shape=(9, 6)))
model.add(Flatten()) ##
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='relu'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping

# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") 

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
# model_path = "".join([filepath, '6_fetch_covtype_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])#, mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
loss : 0.5210703015327454
accuracy : 0.7869246006011963
---------------------------------
LSTM 반영시 값이 더 안좋아졌다
걸린시간 :  125.711 초
loss :  0.7971971035003662
accuracy :  0.6591654419898987
=================================
Conv1D
걸린시간 :  50.612 초
loss :  0.7778472900390625
accuracy :  0.6833730340003967
'''