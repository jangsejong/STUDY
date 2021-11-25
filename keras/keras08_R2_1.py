
'''
 R2 가 먼지 찾아라!!!!
MSE는 전체 데이터의 크기에 의존하기 때문에 서로 다른 두 모델의 MSE만을 비교해서 어떤게 더 좋은 모델인지 판단하기 어렵다는 단점이 있습니다.
이를 해결하기 위한 metric으로는 R2(결정계수)가 있습니다. R2는 1-(RSS/전체 분산)입니다. 
R2는 회귀 모델의 설명력을 표현하는 지표로써, 그 값이 1에 가까울수록 높은 성능의 모델이라고 할 수 있습니다. 
R2의 식에서 분자인 RSS의 근본은 실제값과 예측값의 차이인데, 그 값이 0에 가까울수록 모델이 잘 예측을 했다는 뜻이므로 R2값이 1에 가까워지게 됩니다.

대표적 RSS(단순 오차 제곱 합), MSE(평균 제곱 오차), MAE(평균 절대값 오차)
RSS는 예측값과 실제값의 오차의 제곱합
MSE는 RSS를 데이터의 개수만큼 나눈 값
MAE는 예측값과 실제값의 오차의 절대값의 평균
RMSE와 RMAE : 각각 MSE와 MAE에 루트를 씌운 값

'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss : 12.920090675354004

y_predict = model.predict(x_test)
'''
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
'''

R2 = r2_score(y_test, y_predict)
print(R2) # 0.09155571827545039

import matplotlib.pyplot as plt
plt.scatter(x_test, y_test,color='red')
plt.show()