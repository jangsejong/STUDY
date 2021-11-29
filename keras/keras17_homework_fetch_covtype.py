from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

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

y = to_categorical(y)
#print(y.shape) #
datasets = fetch_covtype()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
#print(x.shape, y.shape)  #

#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=54))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(8, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=200, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=2000, batch_size=54, validation_split=0.2, callbacks=[es]) # batch_size=default 는 32이다.

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

Epoch 493/2000
patience=100
loss : 0.6760594844818115
accuracy : 0.7110315561294556

Epoch 729/2000
patience=200
loss : 0.6630104780197144
accuracy : 0.7179160714149475
'''
