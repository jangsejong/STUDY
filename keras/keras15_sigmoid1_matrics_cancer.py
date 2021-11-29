import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import print_tensor


#1. 데이타 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
#print(datasets)
#print(y_test[:10])

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     (569, 30),(569, )

#print(y) 이진분류, sigmoid 
#print(y[:10])          
#print(np.unique(y))   #[0, 1]

#2. 모델구성

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=30)) # linear 원래값
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(6, activation='linear'))
model.add(Dense(1, activation='sigmoid')) # sigmoid 0,1 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 훈련에 상황만 판단, 데이터가 한쪽에 편중되지 않아야 신뢰도가 높다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=40, mode='min', verbose=1, restore_best_weights=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=10,verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
#y_predict = model.predict(x_test)
#print(y_predict)


'''
epochs=2000, patience=40 일때
Epoch 151/2000
loss : [0.2657780945301056, 0.9122806787490845]

Epoch 126/2000
loss : [0.32039958238601685, 0.8947368264198303] # 2개이상 리스트

epochs=2000, patience=45 일때
Epoch 106/2000
loss : [0.2919485569000244, 0.8947368264198303]

Epoch 151/2000
loss : [0.2657780945301056, 0.9122806787490845]

'''