import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '1_boston_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', verbose=1, mode='min', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

'''
loss :  21.551227569580078
r2스코어 0.7619709144301021
'''