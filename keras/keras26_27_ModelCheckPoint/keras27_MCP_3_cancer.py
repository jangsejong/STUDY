import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.datasets import load_breast_cancer
import time

datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=66)


scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#2. 모델 구성
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=30))
model.add(Dense(30, activation='relu'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '3_cancer_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

#loss :  [0.3958258330821991, 0.9573934674263]