import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.datasets import load_wine
import time

from tensorflow.python.keras.callbacks import ModelCheckpoint

datasets = load_wine()
x = datasets.data 
y = datasets.target

print(x.shape, y.shape)  #(178, 13) (178,)
x = x.reshape(178, 13, 1)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)


# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) 

#2. 모델 구성
model = Sequential()
model.add(LSTM(30, activation='linear', input_shape=(13,1)))
model.add(Dropout(0.5))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") 

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
# model_path = "".join([filepath, '5_wine_', datetime, '_', filename])

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

'''
loss :  [0.26007527112960815, 0.9440000057220459]
------------------------
LSTM 반영시 값이 더 안좋아졌다.
걸린시간 :  3.572 초
loss :  [1.0972484350204468, 0.37599998712539673]
'''
