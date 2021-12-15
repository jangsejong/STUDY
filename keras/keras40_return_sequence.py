import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, GRU
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])




# print(x.shape, y.shape)  #(13, 3) (13,)
#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수



#x = np.array(x, dtype=np.float32)
print(x.shape)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#          train_size = 0.9, shuffle = True) 

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x)
x = scaler.transform(x)
# x_test = scaler.transform(x_test)

x = x.reshape(13, 3, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(300,input_shape=(3, 1), return_sequences=True))  #(n,3,1)> (n,10)
model.add(LSTM(300, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30)) # 마지막은 return_sequence X
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
# model.add(Dropout(0.1))
# model.add(Dense(2, activation='linear'))
# model.add(Dropout(0.1))
model.add(Dense(1))
#model.summary()
'''
LSTM을 사용할 때 훈련 시킨 데이터의 구간 외를 예측하는 것은 오차가 많이 발생

'''

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x, y, epochs=10000, batch_size=10, verbose=1, validation_split=0.1, callbacks=[es])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')


#4. 평가,예측
loss = model.evaluate(x,y)
y_pred = model.predict(x)


results = model.predict([[[50],[60],[70]]])

#results=results.round(0).astype(int)
print(results)

'''
patience= 500
[[80.5892]]

model.add(Dropout(0.1)) 사용시
[[79.8797]]

patience= 200
[[81.77456]]
'''
