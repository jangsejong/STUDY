from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
datasets = load_boston()

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from sklearn import metrics

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV #분류모델이다
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#1 데이터
x = datasets.data
y = datasets.target
print(np.min(x), np.max(x))  #0.0  711.0   

# x = x/711.             #. 안전하다
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  #shuffle 은 기본값 True
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model1 = RandomForestClassifier(oob_score= True, bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators= 15000, n_jobs=None, verbose=0, warm_start=False, random_state=66)

model2 = GradientBoostingClassifier(n_estimators = 15000,random_state=66)

model3 = ExtraTreesClassifier(n_estimators = 15000,random_state =66)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
model4 = HistGradientBoostingClassifier(random_state =66)
model5 = LinearSVC()
model6 = SVC()
model7 = Perceptron()
model8 = KNeighborsClassifier()
model9 = KNeighborsRegressor()
model10 = LogisticRegression()
model11 = LogisticRegressionCV()
model12 = RandomForestRegressor()
model13 = DecisionTreeClassifier()
model14 = DecisionTreeRegressor()
model15 = LinearSVC()
model16 = LinearSVC()



#from lightgbm import LGBMClassifier
#model5 = LGBMClassifier(random_state =66)


voting_model = VotingClassifier(estimators=[ ('RandomForestClassifier', model1), ('GradientBoostingClassifier', model2)
                                            ,('ExtraTreesClassifier', model3),('HistGradientBoostingClassifier', model4)
                                            ,('LinearSVC', model5),('HistGradientBoostingClassifier', model6)
                                            ,('ExtraTreesClassifier', model3),('HistGradientBoostingClassifier', model4)
                                            ,('ExtraTreesClassifier', model3),('HistGradientBoostingClassifier', model4)], voting='hard')

classifiers = [model1, model2,model3,model4]

for classifier in classifiers:
    classifier.fit(x_train,x_test)
    pred = classifier.predict(x_test)
    class_name = classifier.__class__.__name__
    print("============== " + class_name + " ==================")
    num = accuracy_score(y_test, pred)
    num2 = f1_score(y_test, pred)
    print('{0} 정확도: {1}'.format(class_name, num))
    print('{0} F1_Score : {1}'.format(class_name, num2))
    y_pred_ = classifier.predict(y_train)
    # submission['target'] = y_pred_
    # submission.to_csv(path+ str(num) +"_" + class_name + "heart1223_001.csv", index=False)
    
voting_model.fit(x_train, y_train)
pred = voting_model.predict(x_test)




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