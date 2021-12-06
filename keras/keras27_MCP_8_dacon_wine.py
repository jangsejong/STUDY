import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint

#1. 데이터 분석
path = "../../_data/dacon/wine/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['id', 'quality'], axis =1)


le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)         # type column의 white, red > 0,1로 변환


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=8))
model.add(Dense(30, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '8_dacon_wine_', datetime, '_', filename])


es = EarlyStopping(monitor="val_loss", patience=20, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es, mcp])

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test, y_pred))   
rmse = RMSE(y_test, y_pred) 
print("RMSE : ", rmse)