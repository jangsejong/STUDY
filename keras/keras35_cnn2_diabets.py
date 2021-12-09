from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


datasets = load_diabetes()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=66)



scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

n = x_train.shape[0]# 
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255


'''
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #354,13
x_train = x_train_transe.reshape(n,2,2,3) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,2,3)

print(x_train.shape)







scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델 구성
model = Sequential()
model.add(Dense(60, input_dim=10))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(33))
model.add(Dense(26))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Dense(8, activation='relu'))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dropout(0.5))
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
model_path = "".join([filepath, '2_diabetes_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', verbose=1, save_best_only=True, mode='min', filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)

loss :  3934.139892578125
r2스코어 0.3656581301098685
'''