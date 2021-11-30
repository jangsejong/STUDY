import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from pandas import Series, DataFrame

#1. 데이터
path = "./_data/bike/"
train = pd.read_csv(path + "train.csv")# index_col=0,  header=0) # 인덱스 조절하여 1열 삭제, 헤드 조절하여 행 선정
#print(train)
#print(train.shape) #(10886, 12)

test = pd.read_csv(path + "test.csv") #index_col=0,  header=0)
#gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0,  header=0)

#print(test.shape) #(6493, 9)
submit = pd.read_csv(path +'sampleSubmission.csv')
#print(submit.shape) #(6493, 2)

print(type(train)) #<class 'pandas.core.frame.DataFrame'>
#print(train.info())  #object 모든 자료의 최상위형
#print(train.describe()) #std 표준편차  mean, 중위값과 평균값차이


'''

x_train = 
y_train = np.arange
y = pd.get_dummies(y)





#2. 모델구성

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=12))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(2, activation='softmax'))




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=200, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=2000, batch_size=54, validation_split=0.2, callbacks=[es]) # batch_size=default 는 32이다.

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss[0]) #loss : 낮은게 좋다
print('accuracy :', loss[1])
results = model.predict(x_test[:4])
print(y_test[:4])
print(results)

'''