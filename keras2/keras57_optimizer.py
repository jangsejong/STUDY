from pickletools import optimize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense , Dropout, Conv2D, Flatten, Input
import time
import numpy as np


#1. 데이터

x =np.array([1,2,3,4,5,6,7,8,9,10])
y =np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001

# optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
optimizer = Adadelta(learning_rate= learning_rate, rho=0.95, epsilon=1e-7, name='Adadelta')
# optimizer = Adagrad(learning_rate= learning_rate, initial_accumulator_value=0.1, epsilon=1e-7, name='Adagrad')
# optimizer = Adamax(learning_rate= 0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='Adamax')
# optimizer = RMSprop(learning_rate= learning_rate, rho=0.9, momentum=0, epsilon=1e-7, centered=False, name="RMSprop")
# optimizer = SGD(learning_rate= learning_rate, momentum= 0 , nesterov=False, name="SGD")
# optimizer = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='Nadam')


model.compile(loss='mse', optimizer= optimizer)

model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
'''
IndexError: list index out of range
y_pred = model.predict(11)  
'''

y_pred = model.predict([11])  



print('loss :', round(loss, 4), 'lr :', learning_rate, '결과물 :', y_pred)

'''
Adam
loss : 2.1957 lr : 0.0001 결과물 : [[10.690002]]
Adadelta
loss : 2.6178 lr : 0.001 결과물 : [[11.366012]]
Adagrad
loss : 2.3709 lr : 0.001 결과물 : [[10.543584]]
Adamax
loss : 2.384 lr : 0.0001 결과물 : [[11.360344]]
RMSprop
loss : 2.363 lr : 0.001 결과물 : [[11.430712]]
SGD
loss : 2.1758 lr : 0.001 결과물 : [[10.683594]]
Nadam
loss : 2.2257 lr : 0.001 결과물 : [[10.772727]]
'''