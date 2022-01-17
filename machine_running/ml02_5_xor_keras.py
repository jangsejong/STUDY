import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0, 0],[0, 1],[1, 0],[1, 1]]
y_data = [0, 1, 1, 0]
x_data = np.array(x_data)
y_data = np.array(y_data)

x_data=np.asarray(x_data).astype(np.int)
y_data=np.asarray(y_data).astype(np.int)


# x_data=np.asarray(x_data).astype(np.float)
# y_data=np.asarray(y_data).astype(np.float)


#2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC()

model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
# model.add(Dense(8, input_dim=2))#, activation='relu'))
# model.add(Dense(4, input_dim=2, activation='relu'))
# model.add(Dense(2, input_dim=2))#, activation='relu'))
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_data, y_data)

#4. 평가

y_pred = model.predict(x_data)

results = model.evaluate(x_data, y_data)

print(x_data, "의 예측결과 :", y_pred)
# print('matrics_acc : ', results[1])


acc = accuracy_score(y_data, np.round(y_pred,0))

# result = model.predict([1,1])
print('acc 의 예측값 : ', acc)
