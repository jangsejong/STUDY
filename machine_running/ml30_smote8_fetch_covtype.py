import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, load_boston, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import *
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = fetch_covtype(return_X_y=True)
print(x.shape, y.shape) #(581012, 54) (581012,)



# for index, value in enumerate(y):
#     if value == 9 :
#         y[index]==8
#     elif value == 8 :
#         y[index]==8
#     elif value == 7 :
#         y[index]==6
#     elif value == 6 :
#         y[index]==6
#     elif value == 5 :
#         y[index]==1
#     elif value == 4 :
#         y[index]==1
#     elif value == 3 :
#         y[index]==1
#     elif value == 2 :
#         y[index]==1

#     else:
#         y[index] ==1



# newlist = []
# for i in y:
#     # print(i)
#     if i < 4 :
#         newlist +=[0]
#     elif i <= 5 :
#         newlist +=[1]
#     elif i <= 7 :
#         newlist +=[2]        
#     else:
#         newlist +=[3]
        
# y = np.array(newlist)
# print(np.unique(y, return_counts=True))
#print(type(x)) # numpy
x = np.delete(x,[0,6],axis=1)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8)

print("============SMOTE 적용==============")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=66,k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)

#data save
import pickle
pickle.dump(x_train, open('./_save/m30_x_train_save.dat', 'wb'))
pickle.dump(x_test, open('./_save/m30_x_test_save.dat', 'wb'))
pickle.dump(y_train, open('./_save/m30_y_train_save.dat', 'wb'))
pickle.dump(y_test, open('./_save/m30_y_test_save.dat', 'wb'))


# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# from sklearn.model_selection import train_test_split, KFold
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

# #2. 모델
# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parameters = [
#     {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
#     {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]}] # 'eval_metric':['merror']]

# from xgboost import XGBClassifier, XGBRegressor

# model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True)

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# score = model.score(x_test, y_test)
# print('model.score: ', score)
# ''' 
# '''
# #print(model.best_estimator_.feature_importances_)
# ''' 
# '''
# #print(np.sort(model.best_estimator_.feature_importances_))  # 오름차순으로 정렬해주기
# ''' 
# '''
# aaa = np.sort(model.best_estimator_.feature_importances_)  # 오름차순으로 정렬해주기

# print("==============================================================================")
# for thresh in aaa:
#     selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)   
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs = -1)
#     selection_model.fit(select_x_train, y_train, eval_metric='merror')
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
#     print("Thresh = %.3f, n=%d, R2: %2f%%"
#           %(thresh, select_x_train.shape[1], score*100))
  
# #################################################################################
# ''' 

# '''

# """ 
# 기존 model.score)  
# 컬럼 제거 후 model.score)
# smote 적용후
# model.score:  

# if 로 라벨링후
# model.score:
# 컬럼 제거 model.score:  

# smote 적용후
# model.score: 
# 컬럼 제거 model.score:  

# k_neighbors=2
# model.score:  
# k_neighbors=1
# model.score:  

# 라벨링4개
# model.score:  
# """