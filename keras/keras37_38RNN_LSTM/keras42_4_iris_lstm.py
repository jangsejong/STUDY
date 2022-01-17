import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint
import time

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR) # x=(150, 4), y=(150,)
# print(np.unique(y)) # [0, 1, 2]

print(x.shape, y.shape)  #(150, 4) (150,)
x = x.reshape(150, 4, 1)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(LSTM(30, activation='linear', input_shape=(4,1)))
model.add(Dense(30, activation='relu'))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=1000, mode = 'min', verbose = 1) # restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 13, 
                 validation_split = 0.2 , callbacks = [es])#, mcp])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])

'''
loss : 0.060296203941106796
accuracy : 1.0

============================
LSTM
걸린시간 :  25.968 초
loss :  0.09477469325065613
accuracy :  0.9333333373069763

'''