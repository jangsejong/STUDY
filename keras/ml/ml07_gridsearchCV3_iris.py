'''







params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6, 12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2, 8, 16, 20]
            }






'''


import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

datasets = load_iris()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

x_train, x_test,y_train,y_test=train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

#2. 모델구성
# model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True)
model = SVC(C=1, kernel='linear',degree=3)

      
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

print("model.score: ", model.score(x_test, y_test)) #score는 evaluate 개념

y_predict=model.predict(x_test)
print("accuracy_score: ", accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 acc :", accuracy_score(y_test,y_pred_best))

# import pandas as pd

# #########################################################################
# # print(model.cv_results_) #dic
# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# bbb = aaa[['params',',meas_test_scores', 'rank_test_scores', 'split0_test_score']]
# #, 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']]
# print(bbb)

