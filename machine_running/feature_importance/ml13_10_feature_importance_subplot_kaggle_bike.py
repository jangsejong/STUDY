from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper


from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score


#1. 데이터 분석
path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
# y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime'], axis=1)

# x = np.delete(x, 1, 1)
# x = np.delete(x, 1, 1)
# x = np.delete(x,[0,1],axis=1)

#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

# y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
# print(x.shape, y.shape)  #(178, 13) (178,)

#2. 모델구성

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델이다
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

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
# model = DecisionTreeClassifier(max_depth=5, random_state=66)
from xgboost.sklearn import XGBRegressor #회귀
'''
회귀는 잔차(residual : 데이터의 실측치와 모델의 예측치 사이의 차이)가 평균으로 회귀하는것. 잔차가 평균으로 회귀하도록 만들 모델 
잔차가 정규 분포를 띄고 데이터와 상관이 없고 분산이 항상 일정하다면 평균으로 회귀하는 속성을 갖는다.
그리고 이렇게 잔차가 평균으로 회귀하도록 만들 모델을 회귀 모델이라고 합니다.
#Link :https://brunch.co.kr/@gimmesilver/17
'''
from xgboost import XGBClassifier #분류
import xgboost

# model = XGBRegressor()
model1 = DecisionTreeClassifier(max_depth=5)
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()
# model = xgboost()




#3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss',patience=5, mode='min', verbose=1)


# hist = model.fit(x_train, y_train, epochs=10000, batch_size=13, validation_split=0.2, callbacks=[es])

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plt.subplot(1, 4, 1)    
plot_feature_importances_dataset(model1)
plt.subplot(1, 4, 2)    
plot_feature_importances_dataset(model2)
plt.subplot(1, 4, 3)    
plot_feature_importances_dataset(model3)
plt.subplot(1, 4, 4)    
plot_feature_importances_dataset(model4)
plt.show()




# #4. 평가, 예측
# # loss = model.evaluate (x_test, y_test)
# # print('loss :', loss[0]) #loss : 낮은게 좋다
# # print('accuracy :', loss[1])
# results = model1.score(x_test, y_test)


# from sklearn.metrics import accuracy_score
# y_pred = model1.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# # print(y_test[:7])
# print("result : ", results)

# print("accuracy_score : ", acc)

# print(model1.feature_importances_)######

