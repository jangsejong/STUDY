from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
datasets= fetch_covtype()
'''
print(datasets.DESCR)
The samples in this dataset correspond to 30횞30m patches of forest in the US,
collected for the task of predicting each patch's cover type,
i.e. the dominant species of tree.
There are seven covertypes, making this a multiclass classification problem.
Each sample has 54 features, described on the
`dataset's homepage <https://archive.ics.uci.edu/ml/datasets/Covertype>`__.
Some of the features are boolean indicators,
while others are discrete or continuous measurements.
'''
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(581012, 54) (581012,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [1 2 3 4 5 6 7]

#y = to_categorical(y)  #output 8
#print(y.shape) #
'''
from sklearn.preprocessing import OneHotEncoder
k = OneHotEncoder(sparse=False) # sparse=True가 디폴트이면 이는 Matrix를 반환한다. 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
y = k.fit_transform(y.reshape(-1, 1)) # -1 번째 열을 1 번째 열로 변환
'''
import pandas as pd
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
#print(x.shape, y.shape)  


#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=54))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='relu'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=2000, batch_size=108, validation_split=0.2, callbacks=[es]) # batch_size=default 는 32이다.

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss[0]) #loss : 낮은게 좋다
print('accuracy :', loss[1])
results = model.predict(x_test[:2])
print(y_test[:2])
print(results)

'''
Epoch 99/1000
patience=20
loss : 0.6677718758583069
accuracy : 0.7189056873321533

============================

# MinMaxScaler
loss : 0.6607382297515869
accuracy : 0.7211604118347168

layer에 relu 반영시 값이 좋아졌다
loss : 0.5210703015327454
accuracy : 0.7869246006011963

'''
'''
# MinMaxScaler                       #layer 에 relu 반영시

loss : 0.6607382297515869            loss : 0.5210703015327454  
accuracy : 0.7211604118347168        accuracy : 0.7869246006011963

# StandardScaler
loss : 0.6613174676895142           loss : 8.482542037963867       
accuracy : 0.7198092937469482         r2score : 0.8973270448642088  

# RobustScaler
loss : 0.6603913903236389            loss : 9.703550338745117   
accuracy : 0.7201879620552063        r2score : 0.8825479273496296   

# MaxAbsScaler
loss : 0.6628473401069641             loss : 9.844630241394043   
accuracy : 0.7198178768157959        r2score : 0.880840280687609   


'''