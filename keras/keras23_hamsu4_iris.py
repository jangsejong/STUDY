from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
datasets= load_iris()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)    #(150, 4) (150,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  #[0, 1, 2]
y = to_categorical(y)
#print(y.shape) #(150, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

#print(x.shape, y.shape)  


#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

input1 = Input(shape=(4,))
dense1 = Dense(30)(input1)
dense2 = Dense(30, activation='relu')(dense1)
dense3 = Dense(18)(dense2)
dense4 = Dense(6)(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(2)(dense5)
ouput1 = Dense(3, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)

'''
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=4))
model.add(Dense(30, activation='relu'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))
'''
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=2000, batch_size=10, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
#print('loss :', loss[0]) #loss : 낮은게 좋다
#print('accuracy :', loss[1])
results = model.predict(x_test[:7])

#print(y_test[:7])
#print(results)

'''
loss : 0.060296203941106796
accuracy : 1.0

============================

# MaxAbsScaler
loss: 0.0492 - accuracy: 1.0000

# layer 에 relu 반영시 값이 좋아졌다
loss: 0.0442 - accuracy: 1.0000


# MinMaxScaler                       #layer 에 relu 반영시 epo값 끝까지 가는..

loss: 0.0524 - accuracy: 1.0000      loss: 0.0480 - accuracy: 0.9896

# StandardScaler
loss: 0.0479 - accuracy: 0.9667      loss: 1.8626e-08 - accuracy: 1.0000

# RobustScaler
loss: 0.0593 - accuracy: 1.0000      loss: 0.0370 - accuracy: 0.9792

# MaxAbsScaler
loss: 0.0492 - accuracy: 1.0000      loss: 0.0442 - accuracy: 1.0000 

#함수형모델 사용시
# StandardScaler
loss: 1.7385e-08 - accuracy: 1.0000 - val_loss: 1.7868 - val_accuracy: 0.9167
'''