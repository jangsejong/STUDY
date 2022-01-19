import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import *
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

datasets = load_boston()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

x_train, x_test, y_train, y_test=train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


# n_splits=3
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters=[
    {"randomforestclassifier__max_depth":[6, 8, 10]}, #3
    {"randomforestclassifier__min_samples_leaf":[3, 5, 7], "randomforestclassifier__min_samples_split":[3, 5, 10]},   #9
]   #총  12개

# parameters=[
#     {"rf__max_depth":[6, 8, 10]}, #3
#     {"rf__min_samples_leaf":[3, 5, 7], "rf__min_samples_split":[3, 5, 10]},   #9
# ]   #총  12개


#2. 모델구성

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA

pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier()) 
# pipe = Pipeline([("Scaler", MinMaxScaler()),("rf", RandomForestClassifier())])

model = GridSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측

# print("최적의 매개변수 : ", model.best_estimator_)
# print("최적의 파라미터 : ", model.best_params_)

# print("best_score_: ", model.best_score_)
# print("model.score: ", model.score(x_test, y_test)) #score는 evaluate 개념

y_predict=model.predict(x_test)
print("accuracy_score: ", accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 acc :", accuracy_score(y_test,y_pred_best))

# import pandas as pd

#########################################################################
'''


print(model.cv_results_) #dic
aaa = pd.DataFrame(model.cv_results_)
print(aaa)

bbb = aaa[['params',',meas_test_scores', 'rank_test_scores', 'split0_test_score']]
#, 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']]
print(bbb)
'''