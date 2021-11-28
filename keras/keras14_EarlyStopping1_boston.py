from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(55, input_dim=13))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=13, validation_split=0.3, callbacks=[es])   #통상적으로 효율성이 좋다. 성능이 낮아질 경우 모델이 좋지 않다.
print(hist)


end = time.time() - start
print("걸린시간 : ", round(end, 3))

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) 

'''
print("========================")
print(hist)
print("========================")
print(hist.history)
print("========================")
print(hist.history['loss'])
print("========================")
print(hist.history['val_loss'])
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()


'''
#1  Epoch 10000 ,patience= 50 
Epoch 376/10000     
loss : 20.66997718811035
r2score : 0.749809940396891
Epoch 326/10000 loss: 29.6490 - val_loss: 34.8927

Epoch 503/10000     
loss : 23.355093002319336
r2score : 0.7173092205607651
Epoch 453/10000 loss: 30.2731 - val_loss: 34.1916

#2 Epoch 500 일때
Epoch 486/500      
loss : 22.983949661254883
r2score : 0.7218015469957169
Epoch 436/500  loss: 27.0648 - val_loss: 33.3602
--------
Epoch 459/500       
loss : 20.585783004760742
r2score : 0.7508290498437922
Epoch 409/500  loss: 28.9417 - val_loss: 34.0244
--------
Epoch 397/500
loss : 19.313711166381836
r2score : 0.766226230291853
Epoch 347/500  loss: 27.8140 - val_loss: 34.9563
==================================================
#1 patience=100 ,patience=100
Epoch 723/10000
loss : 18.624523162841797
r2score : 0.7745681857837043
Epoch 623/10000   loss: 29.0952 - val_loss: 30.6227
---------
Epoch 595/10000
loss : 18.361373901367188
r2score : 0.7777533562632721
Epoch 495/10000   loss: 29.0117 - val_loss: 33.5989
---------
Epoch 486/10000
loss : 19.056377410888672
r2score : 0.7693410056538847
Epoch 386/10000   loss: 31.4629 - val_loss: 34.4500

#2 patience=100 에 Epoch 800 일때


'''