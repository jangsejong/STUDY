from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=4))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=250, mode='min', verbose=1)


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

# MinMaxScaler
loss: 0.0524 - accuracy: 1.0000

# StandardScaler
loss: 0.0479 - accuracy: 0.9667

# RobustScaler
loss: 0.0593 - accuracy: 1.0000

# MaxAbsScaler
loss: 0.0492 - accuracy: 1.0000

'''