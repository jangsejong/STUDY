import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer
import time
from sklearn import metrics


path = "D:\\Study\\_data\\dacon\\heart\\"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_submission.csv")

x = train.drop(['id','target'], axis =1)
y = train['target']

#print(train.shape, test_file.shape)           # (151, 15) (152, 14)

x = x.drop(['trestbps','restecg','sex'],axis =1)
test_file =test_file.drop(['id','trestbps','restecg','sex'],axis =1)

y = y.to_numpy()
x = x.to_numpy()
test_file = test_file.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=66)


scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#2. 모델 구성
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=10))
model.add(Dense(30, activation='relu'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '_heart_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.1, callbacks=[es, mcp])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)


def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score
#print('F1_score: ', f1_score( x_test, y_predict ))
f1=f1_score(y_test, y_predict)
print('f1_score :  ', f1)
#loss :   [0.2044006884098053, 0.9032257795333862]