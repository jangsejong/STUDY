import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) #(3,10)

y = np.array([11,12,13,14,15,16,17,18,19,20])

#[[10, 1.3, 1]] 결과값 예측

x = np.transpose(x)
print("x=",x)


#2. 모델구성
model = Sequential()
model.add(Dense(25, input_dim=3))
model.add(Dense(250))          #하이퍼 파라미트 튜닝
model.add(Dense(1000))
model.add(Dense(250))
model.add(Dense(25))
model.add(Dense(1))




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') #mne
model.fit(x, y, epochs=800, batch_size=1)



#4. 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([[11, 1.2, 0]])
print('[11, 1.2, 0]의 예측값 : ', y_predict)

'''
loss :  3.637978807091713e-12
[10, 1.3, 1]의 예측값 :  [[20.000002]]
'''
