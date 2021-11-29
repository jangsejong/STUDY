import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이타 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
#print(datasets)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     (569, 30),(569, )

#print(y) 이진분류, sigmoid
print(np.unique(y))   #[0, 1]

#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=30))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=250, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=10, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)



'''
epochs=500, patience=100 일때

Epoch 139/500
loss : 0.18464960157871246
Epoch 39/500      loss: 0.2485 - val_loss: 0.0360

Epoch 472/500
loss : 0.11719817668199539
Epoch 372/500     loss: 0.1071 - val_loss: 0.0402  

Epoch 160/500
loss : 0.18607252836227417
Epoch 60/500      loss: 0.2295 - val_loss: 0.0481
-----------------
epochs=500, patience=10 일때
Epoch 39/500
loss : 0.23204709589481354
Epoch 29/500        loss: 0.3356 - val_loss: 0.0635

Epoch 32/500
loss : 0.3450007438659668
Epoch 22/500        loss: 0.5705 - val_loss: 0.0730
-----------------

epochs=2000, patience=250 일때
Epoch 672/2000
loss: 0.1178 - val_loss: 0.0455
Epoch 422/2000       loss: 0.1011 - val_loss: 0.0370

Epoch 335/2000
loss : 0.17167440056800842
Epoch 135/2000      loss: 0.2580 - val_loss: 0.0433
'''