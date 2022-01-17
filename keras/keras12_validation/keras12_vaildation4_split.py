from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8125, shuffle=True, random_state=66) 
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.3) #validation 사용시 성능이 더 좋아진다.
         # validation_data = (x_val, y_val))
        
'''
loss : 3.031649096259942e-13
17의 예측값:  [[17.]] #숫자17?
'''

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :

y_predict = model.predict([17])
print("17의 예측값: ", y_predict)