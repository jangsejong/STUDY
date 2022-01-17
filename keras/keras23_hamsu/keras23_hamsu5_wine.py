from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
datasets= load_wine()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(178, 13) (178,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

y = to_categorical(y)
#print(y.shape) #(178, 3)
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
#print(x.shape, y.shape)  #(178, 13) (178,)


#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(18)(dense2)
dense4 = Dense(6, activation='relu')(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(2)(dense5)
ouput1 = Dense(3, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=30, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=10000, batch_size=13, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss[0]) #loss : 낮은게 좋다
print('accuracy :', loss[1])
results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''

patience=300
Epoch 3731/10000
loss : 0.03596608713269234
accuracy : 0.9722222089767456

============================
# MaxAbsScaler
loss : 0.0015572120901197195
accuracy : 1.0

relu 반영시 값이 더 좋아졌다
loss : 1.3410724477580516e-06
accuracy : 1.0

'''
'''
# MinMaxScaler                         #layer 에 relu 반영시

loss : 0.35055699944496155             loss : 7.262266444740817e-05  
accuracy : 0.9722222089767456          accuracy : 1.0 

# StandardScalers : 
loss : 0.3546077311038971              loss : 0.32601988315582275     
accuracy  : 0.9722222089767456         accuracy : 0.9722222089767456
 
# RobustScaler
loss : 0.5228054523468018              loss : 0.4000712037086487  
accuracy  : 0.9722222089767456         accuracy : 0.9722222089767456 

# MaxAbsScaler
loss : 0.0015572120901197195           loss : 1.3410724477580516e-06
accuracy  : 1.0                        accuracy : 1.0

#함수형모델 사용시
# MaxAbsScaler
loss: 1.2659e-08 - accuracy: 1.0000
'''