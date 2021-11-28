from sklearn.datasets import load_boston, load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential()
model.add(Dense(60, input_dim=10))
model.add(Dense(50))
model.add(Dense(41))
model.add(Dense(33))
model.add(Dense(26))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=800, batch_size=10, validation_split=0.2, callbacks=[es])   #통상적으로 효율성이 좋다. 성능이 낮아질 경우 모델이 좋지 않다.


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
#1  EEpoch 10000 ,patience= 50 일때
Epoch 133/10000 
loss : 2544.6337890625
r2score : 0.5455030264268246
Epoch 83/10000 loss: 3067.8652 - val_loss: 3334.2358

Epoch 112/10000    
loss : 2542.4296875
r2score : 0.5458966554362995
Epoch 62/10000 3159.1282 - val_loss: 3335.4302

#2 patience= 200 일때
Epoch 327/10000   
loss : 2504.949951171875
r2score : 0.5525909208483328
Epoch 227/10000 loss: 3186.4937 - val_loss: 3820.7900
--------
Epoch 346/10000     
loss : 2673.100830078125
r2score : 0.5225575073615305
Epoch 246/10000  loss: 3067.9976 - val_loss: 3401.9189
--------
poch 372/10000
loss : 2523.278076171875
r2score : 0.5493172963046924
Epoch 272/10000  loss: 3276.1885 - val_loss: 3411.8374
==================================================
#1 patience=400 일때
Epoch 691/10000
loss : 2652.735595703125
r2score : 0.5261949120936278
Epoch 291/10000   loss: 3106.4209 - val_loss: 3311.7910
---------
Epoch 592/10000
loss : 2540.089599609375
r2score : 0.5463146344806651
Epoch 192/10000   loss: 3136.5847 - val_loss: 3320.9036ㅍ
---------
Epoch 620/10000
loss : 2515.3798828125
r2score : 0.5507280039247142
Epoch 420/10000   loss: 3042.0508 - val_loss: 3401.4302

#2 patience=100 에 Epoch 800 일때

Epoch 385/800
loss : 2518.140869140625
r2score : 0.5502349392445551
Epoch 285/800     loss: 3233.4338 - val_loss: 3322.7322


Epoch 246/800
loss : 2560.132568359375
r2score : 0.542734769843649
Epoch 146/800     loss: 3137.3328 - val_loss: 3318.2231 

'''