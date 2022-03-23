from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

#1. 데이터
datasets= load_iris()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(178, 13) (178,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

# y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
# print(x.shape, y.shape)  #(178, 13) (178,)

#2. 모델구성

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델이다
from sklearn.ensemble import RandomForestClassifier

# model = Sequential()
# model.add(Dense(30, activation='linear', input_dim=13))
# model.add(Dense(30, activation='linear'))
# model.add(Dense(18, activation='linear'))
# model.add(Dense(6, activation='linear'))
# model.add(Dense(4, activation='linear'))
# model.add(Dense(2, activation='linear'))
# model.add(Dense(3, activation='softmax'))

# model = LinearSVC()
# model = Perceptron()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = RandomForestClassifier()
model = KNeighborsClassifier()




#3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=5, mode='min', verbose=1)


# hist = model.fit(x_train, y_train, epochs=10000, batch_size=13, validation_split=0.2, callbacks=[es])

model.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate (x_test, y_test)
# print('loss :', loss[0]) #loss : 낮은게 좋다
# print('accuracy :', loss[1])
results = model.score(x_test, y_test)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
# print(y_test[:7])
print("result : ", results)
print("accuracy_score : ", acc)



'''
model = LinearSVC()
result :  0.9166666666666666
accuracy_score :  0.9166666666666666

# model = Perceptron()
result :  0.6388888888888888
accuracy_score :  0.6388888888888888

# model = SVC()
result :  0.6944444444444444
accuracy_score :  0.6944444444444444

# model = KNeighborsClassifier()
result :  0.6944444444444444
accuracy_score :  0.6944444444444444

# model = LogisticRegression()
result :  0.9722222222222222
accuracy_score :  0.9722222222222222

# model = RandomForestClassifier()
result :  1.0
accuracy_score :  1.0

# model = KNeighborsClassifier()
result :  0.6944444444444444
accuracy_score :  0.6944444444444444

'''