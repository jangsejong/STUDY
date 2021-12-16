from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.datasets import load_diabetes
import time


datasets = load_diabetes()
x = datasets.data 
y = datasets.target

print(x.shape, y.shape)  #(442, 10) (442,)
x = x.reshape(442, 10, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)
'''
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
'''
#2. 모델 구성
model = Sequential()
model.add(Conv1D(60,2, input_shape=(10,1)))
model.add(Flatten()) ##
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(33))
model.add(Dense(26))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Dense(8, activation='relu'))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '2_diabetes_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=100, mode = 'min', verbose = 1) # restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 13, 
                 validation_split = 0.2 , callbacks = [es])#, mcp])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)
'''
loss :  3934.139892578125
r2스코어 0.3656581301098685

LSTM 반영시 값이 안좋아졌다 
걸린시간 :  5.625 초
loss :  7012.57275390625
r2스코어 -0.13070911280986341
======================
걸린시간 :  2.984 초
loss :  11659.830078125
r2스코어 -0.8800341758826651
'''