import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이타 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
#print(datasets)
# print(x.shape, y.shape)     (569, 30),(569, )
#print(y) 이진분류, sigmoid
#print(np.unique(y))   #[0, 1]


#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=30))
model.add(Dense(30, activation='relu'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)



'''
epochs=1000, patience=100 일때

# MaxAbsScaler
Epoch 275/1000
loss : 0.07751503586769104
Epoch 175/1000
loss: 0.0658 - val_loss: 0.0272

#layer 에 relu 반영시 
Epoch 952/1000
loss : 0.5388984680175781
Epoch 852/1000
loss: 3.4874e-10 - val_loss: 0.3796


# MinMaxScaler                       #layer 에 relu 반영시

loss: 0.0662 - val_loss: 0.0291         loss: 0.0106 - val_loss: 0.0060

# StandardScaler
loss: 0.0637 - val_loss: 0.0325         loss: 0.0407 - val_loss: 0.0600 

# RobustScaler
loss: 0.0541 - val_loss: 0.0318         loss: 0.0478 - val_loss: 0.0919

# MaxAbsScaler
loss: 0.0658 - val_loss: 0.0272         loss: 3.4874e-10 - val_loss: 0.3796


'''