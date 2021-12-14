import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)  #(13, 3) (13,)

#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수

x = x.reshape(13, 3, 1)

#x = np.array(x, dtype=np.float32)
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.9, shuffle = True) 


model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(3, 1)))
model.add(Dense(8, activation='linear'))
model.add(Dropout(0.1))
# model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dropout(0.1))
# model.add(Dense(2, activation='linear'))
# model.add(Dropout(0.1))
model.add(Dense(1))


#3. 컴파일, 훈련



model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 200 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.1, callbacks=[es])



#4. 평가,예측
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)


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
# import matplotlib.pyplot as plt


# y_pred = model.predict(x_test, batch_size=13)
# plt.scatter(y_test, y_pred)
# plt.xlabel("Price Index: $Y_i$")
# plt.ylabel("Predicted price Index: $\hat{Y}_i$")
# plt.title("Prices vs Predicted price Index: $Y_i$ vs $\hat{Y}_i$")