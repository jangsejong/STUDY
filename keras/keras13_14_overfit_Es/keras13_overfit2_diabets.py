from sklearn.datasets import load_boston, load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터

#datasets = load_boston()
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
start = time.time()
hist = model.fit(x_train, y_train, epochs=235, batch_size=10, validation_split=0.3)   #통상적으로 효율성이 좋다. 성능이 낮아질 경우 모델이 좋지 않다.
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
loss : 2620.906982421875
r2score : 0.5318798460632301
'''