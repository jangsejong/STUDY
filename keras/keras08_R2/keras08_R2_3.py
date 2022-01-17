import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(700))
model.add(Dense(900))
model.add(Dense(700))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=900, batch_size=5)


#4. 평가, 예측
loss = model.evaluate (x, y)
print('loss :', loss) #loss :

y_predict = model.predict(x)


R2 = r2_score(y, y_predict)
print(R2) # 


'''
loss : 0.3800000846385956
0.8100000143048632
'''