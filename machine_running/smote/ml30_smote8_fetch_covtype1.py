import numpy as np
import pandas as pd
import sys
import platform

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, load_boston, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import *
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier

#1. 데이터
import joblib
x_train = joblib.load(open('../_save/m30_x_train_save.dat', 'rb'))
x_test = joblib.load(open('../_save/m30_x_test_save.dat', 'rb'))
y_train = joblib.load(open('../_save/m30_y_train_save.dat', 'rb'))
y_test = joblib.load(open('../_save/m30_y_test_save.dat', 'rb'))


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
# scaler = PolynomialFeatures()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

lgbm_wrapper = LGBMClassifier(learning_rate=0.01, max_depth=120,
                                             n_estimators=1000,
                                             num_leaves=240)

# evals = [(x_test, y_test)]
# lgbm_wrapper.fit(x_train, y_train, early_stopping_rounds=5, eval_metric="multi_logloss",     #one of [None, 'micro', 'macro', 'weighted'].
#                  eval_set=evals, verbose=True)

# import pickle
# lgbm_wrapper = pickle.load(open('../_save/m23_cov_save1.dat', 'rb'))

preds = lgbm_wrapper.predict(x_test)
pred_proba = lgbm_wrapper.predict_proba(x_test)[:, 1]




from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

# def get_clf_eval(y_test, pred=preds, pred_proba=pred_proba):
#     confusion = confusion_matrix( y_test, pred)
#     accuracy = accuracy_score(y_test , pred)
#     precision = precision_score(y_test , pred, average="macro")
#     recall = recall_score(y_test , pred, average="macro")
#     f1 = f1_score(y_test,pred, average="macro")
#     # ROC-AUC 추가 
#     roc_auc = roc_auc_score(y_test, pred_proba)
#     print('오차 행렬')
#     print(confusion)
#     # ROC-AUC print 추가
#     print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
#     F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    
# get_clf_eval(y_test, preds, pred_proba)
f1 = f1_score(y_test,preds, average="micro")
print(f1)
# from lightgbm import plot_importance
# import matplotlib.pyplot as plt
#저장

# pickle.dump(lgbm_wrapper, open('../_save/m23_cov_save1.dat', 'wb'))
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(lgbm_wrapper, ax=ax)
'''
import numpy as np
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

x_train = np.load('../save/_save/smote_x.npy')
y_train = np.load('../save/_save/smote_y.npy')

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9,
#               enable_categorical=False, eval_metric='merror', gamma=0,
#               gpu_id=-1, importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=5,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=1000, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', predictor= 'gpu_predictor', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None 
# )  

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("model.score : ", round(score, 4))

# y_predict = model.predict(x_test)
# print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
# print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))

# path = '../save/_save/'

# import joblib
# joblib.dump(model, path +"weight_save.dat")
 
'''
'''
model.score :  0.9387
accuracy score:  0.9387
f1_score :  0.9354
'''
'''
path = '..\save\_save'
import joblib
model = joblib.load(path +"weight_save.dat")

score = model.score(x_test, y_test)

print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))

'''