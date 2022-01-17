import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, GRU
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
a = np.array(range(1,101))
# print(a) #[ 1  2  3  4  5  6  7  8  9 10 ... 100]
# print(a.shape) #(10,)
x_predict = np.array(range(96, 106))
print(x_predict) #[ 96  97  98  99 100 101 102 103 104 105]

size = 5   # x 4개 ,y 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)- size + 1 ):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:,:-1] 
y = dataset[:, -1] 

x_predict = split_x(x_predict,size)
x_predict_x = x_predict[:,:-1] 
print(x_predict_x)
print(x_predict_x.shape)


x_predict_x = x_predict_x.reshape(6,4,1)

# scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# #scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# # x_test = scaler.transform(x_test)
# x = x.reshape(13, 3, 1)
# x = dataset[:,:-1].reshape(96, 4, 1)  


#2. 모델구성
model = Sequential()
model.add(LSTM(30, input_shape=(4, 1), return_sequences=True))  
# model.add(LSTM(30, return_sequences=True))
# model.add(LSTM(30, return_sequences=True))
model.add(LSTM(10)) # 마지막은 return_sequence X
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
# model.add(Dropout(0.1))
# model.add(Dense(2, activation='linear'))
# model.add(Dropout(0.1))
model.add(Dense(4))
#model.summary()


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 2 , mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x_predict_x, y, epochs=10, batch_size=10, verbose=1, validation_split=0.1, callbacks=[es])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')


#4. 평가,예측
loss = model.evaluate(x_predict,y)
y_pred = model.predict(x_predict)


y_predict = model.predict(x_predict_x)
# y_pred = y_predict.reshape(10,)
#results=results.round(0).astype(int)
print(y_pred)

