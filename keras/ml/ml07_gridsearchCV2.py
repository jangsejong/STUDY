
from tabnanny import verbose
import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import seaborn as sns
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델이다
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

'''
#1. 데이터
datasets= load_iris()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold, GridSearchCV


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=66)


# scoring = 'neg_root_mean_squared_error'
scoring = 'cross_val_score'

n_splits =5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


models = []
models.append(('LR',LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('CART',DecisionTreeRegressor()))
models.append(('EN',ElasticNet()))
models.append(('SVM',SVR()))
models.append(('RFR',RandomForestRegressor()))
models.append(('XGBR',XGBRegressor()))
models.append(('LGBMR',LGBMRegressor()))
models.append(('AdaR',AdaBoostRegressor()))
models.append(('Cat',CatBoostRegressor(verbose=False)))
models.append(('Xtree',ExtraTreesRegressor()))

results =[]
names = []
for name, model in models:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
#   cv_results = cross_val_score(model,x,y
#                                ,cv= kfold,scoring=scoring)
#   results.append(cv_results)
#   names.append(name)
#   msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
#   print(msg)
  
  #standardization

pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO',Pipeline([('Scaler',preprocessing.StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',preprocessing.StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',preprocessing.StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledEN',Pipeline([('Scaler',preprocessing.StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledSVM',Pipeline([('Scaler',preprocessing.StandardScaler()),('SVM',SVR())])))
pipelines.append(('ScaledRFR',Pipeline([('Scaler',preprocessing.StandardScaler()),('RFR',RandomForestRegressor())])))
pipelines.append(('ScaledXGBR',Pipeline([('Scaler',preprocessing.StandardScaler()),('XGBR',XGBRegressor())])))
pipelines.append(('ScaledLGBMR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LGBMR',LGBMRegressor())])))
pipelines.append(('ScaledAdaR',Pipeline([('Scaler',preprocessing.StandardScaler()),('AdaR',AdaBoostRegressor())])))
pipelines.append(('ScaledCat',Pipeline([('Scaler',preprocessing.StandardScaler()),('Cat',CatBoostRegressor(verbose=False))])))
pipelines.append(('ScaledXtree',Pipeline([('Scaler',preprocessing.StandardScaler()),('Xtree',ExtraTreesRegressor())])))


from sklearn.metrics import *

#   cv_results = score(model,x,y
#                                ,cv= kfold,scoring=scoring)
#   results_scaled.append(cv_results)
#   names_scaled.append(name)
#   msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
#   print(msg)
  
  
params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6, 12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2, 8, 16, 20]
            }
# params = {"c":[1, 10, 100,1000],"kernel":["linear"], "degree":[3,4,5]}, 
# {"c":[1, 10, 100],"kernel":["rbf"], 'gamma':[0.01, 0.001, 0.0001]}, #9
# {"c":[1, 10, 100,1000],"kernel":["sigmoid"],"gamma":[0.01, 0.001, 0.0001],'degree':[3,4]}                      #24


model = GridSearchCV(SVR(), params, cv=kfold, verbose=1)


model1 = model.fit(x_train, y_train)

# grid = GridSearchCV(estimator= model1, param_grid = params,scoring= 'neg_root_mean_squared_error',cv=kfold )
# grid_result = grid.fit(x_train,y_train)
print("최적의 매개변수 :", model1.best_estimator_)
print("최적의 파라미터 :", model1.best_params_)
print("BEST_SCORES_ :", model1.best_score_)
print("model1.score :",model1.score(x_train, y_train))

y_pred = model1.predict(x_test)
print("acc_scores : ", accuracy_score(y_test, y_pred))
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
