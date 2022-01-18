'''







params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6, 12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2, 8, 16, 20]
            }






'''


import numpy as np
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import fetch_covtype
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

datasets = fetch_covtype()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

num_folds= 10
seed = 7
scoring = 'neg_root_mean_squared_error'
X_all = x
y_all =y

X_train, X_valid, y_train, y_valid=train_test_split(X_all,y_all, shuffle=True, random_state=66, train_size=0.8)

#2. 모델구성
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.svm import SVR
from sklearn.ensemble import *
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

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
# model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True)
# model = SVC(C=1, kernel='linear',degree=3)

results =[]
names = []
for name, model in models:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
  cv_results = cross_val_score(model,X_train,y_train
                               ,cv= kfold,scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
  print(msg)



############

from sklearn import preprocessing
  
  
  
scaler = preprocessing.StandardScaler().fit(X_all)
scaled_X = scaler.transform(X_all)
params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6, 12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2, 8, 16, 20]
            }
model = RandomForestRegressor()
kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
grid = GridSearchCV(estimator= model, param_grid = params,scoring= 'neg_root_mean_squared_error',cv=kfold )
grid_result = grid.fit(scaled_X,y_all)

print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))  
  
  
  
  
params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6,12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2,4,8, 16]
            }
model =ExtraTreesRegressor()
kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
grid = GridSearchCV(estimator= model, param_grid = params, scoring= 'neg_root_mean_squared_error',cv=kfold )
grid_result = grid.fit(X_all,y_all)  
  
  

print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))

  
  
from sklearn.metrics import mean_squared_error
import math   
  
  
errors = []
pred_valid=[]
pred_test = []  
test =  y
  
# scaler = preprocessing.StandardScaler().fit(X_train)
# scaled_X_train = scaler.transform(X_train)
# scaled_X_valid = scaler.transform(X_valid)
# scaled_X_test = scaler.transform(test) 
  
  
  
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_valid = lasso.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, lasso_valid))
errors.append(('Lasso',rmse))
pred_valid.append(('Lasso',lasso_valid))
lasso_test = lasso.predict(test)
pred_test.append(('Lasso',lasso_test))  
  
  
LR =LinearRegression()
LR.fit(X_train,y_train)
lr_valid = LR.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, lr_valid))
errors.append(('LR',rmse))
pred_valid.append(('LR',lr_valid))
lr_test = LR.predict(test)
pred_test.append(('LR',lr_test))  
  
  
RF =RandomForestRegressor(max_depth= 24, min_samples_leaf= 12, min_samples_split= 16, n_estimators= 40)
RF.fit(X_train,y_train)
rf_valid = RF.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, rf_valid))
errors.append(('RF',rmse))
pred_valid.append(('RF',rf_valid))
rf_test = RF.predict(test)
pred_test.append(('RF',rf_test))  
  
ET =ExtraTreesRegressor(max_depth=24, min_samples_leaf= 12, min_samples_split= 16, n_estimators= 40)
ET.fit(X_train,y_train)
et_valid = ET.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, et_valid))
errors.append(('ET',rmse))
pred_valid.append(('ET',et_valid))
et_test = ET.predict(test)
pred_test.append(('ET',et_test))  
  
CAT = CatBoostRegressor(iterations=10000,random_state=66
           ,eval_metric="RMSE")
CAT.fit(X_train,y_train, eval_set=[(X_valid,y_valid)],early_stopping_rounds=30
        ,verbose=1000 )
cat_valid = CAT.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, cat_valid))
errors.append(('CAT',rmse))
pred_valid.append(('CAT',cat_valid))
cat_test = CAT.predict(test)
pred_test.append(('CAT',cat_test))


  
  
for name, error in errors:
      print("{} : {}".format(name,error)) 
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  val+= (0.2* pred)
math.sqrt(mean_squared_error(y_valid, val))  
  
  
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    val+= (0.25* pred)
math.sqrt(mean_squared_error(y_valid, val)) 
  
  
  
test_val= np.zeros(test.shape[0])
for name, pred in pred_test:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    test_val+= (0.25* pred)
      
#3. 훈련
# model.fit(x_train, y_train)

#4. 평가, 예측

print("model.score: ", model.score(X_valid, y_valid)) #score는 evaluate 개념

y_predict=model.predict(X_valid)
print("accuracy_score: ", accuracy_score(y_valid, y_predict))

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

