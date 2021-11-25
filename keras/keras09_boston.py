from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
datasets = load_boston()
#1. 데이터
x = datasets.data
y = datasets.target
'''
print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)
'''


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=13))
model.add(Dense(11))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(65))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=600, batch_size=13)


#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score :', r2) # 
