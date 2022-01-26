import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0, 0],[0, 1],[1, 0],[1, 1]]
y_data = [0, 0, 0, 1]

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가

y_pred = model.predict(x_data)
results = model.score(x_data, y_data)

acc = accuracy_score(y_data, y_pred)
# print(y_test[:7])
print("model score : ", results)
print("accuracy_score : ", acc)
