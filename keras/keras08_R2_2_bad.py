import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# R2를 음수가 아닌 0.5 이하로 만들것
# 데이터 건들지 말것!
# 레이어는 인풋, 아웃풋 포함 6개 이상
# batch_size = 1
# epochs 는 100 이상
# 히든레이어의 노드는 10개 이상 1000개 이하
# train 70%

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))


#2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=1))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(10))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(10))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :

y_predict = model.predict(x_test)


R2 = r2_score(y_test, y_predict)
print(R2) # 

import matplotlib.pyplot as plt
plt.scatter(x_test, y_test,color='red')
plt.show()

'''

'''