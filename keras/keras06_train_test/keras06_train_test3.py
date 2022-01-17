import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# 랜덤으로

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66) # 랜덤스테이트 고정랜덤추출, 랜덤난수
#x_test = ramdom.x
#y_train = ramdom.y
#y_test = ramdom.y

#print(x_train)
print(x_test)
#print(y_train)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(18, input_dim=1))
model.add(Dense(80))          #하이퍼 파라미트 튜닝
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(190))
model.add(Dense(160))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #mne
model.fit(x_train, y_train, epochs=2000, batch_size=1)


#4. 평가,예측
loss = model.evaluate(x_train, y_train)
print('loss : ', loss)
result = model.predict([101])
print('[101]의 예측값 : ', result)

'''
loss :  6.634127913685006e-09
[101]의 예측값 :  [[102.00014]]
'''