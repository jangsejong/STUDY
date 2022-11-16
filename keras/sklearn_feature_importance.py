# permutation feature importance
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import shap
import eli5
from eli5.sklearn import PermutationImportance

import matplotlib.pyplot as plt

import warnings
import gc
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
 
def clf(x, y, col_list):
# random forest
    rfc = RandomForestClassifier(max_depth=12, random_state=66, criterion = "entropy",n_estimators = 3, max_features='auto')
    rfc.fit(x, y)
    result_rfc = permutation_importance(rfc, x,y , n_repeats=10, random_state=66, n_jobs=2)
    sorted_idx_rfc = result_rfc.importances_mean.argsort()
    importances_rf = pd.DataFrame(result_rfc.importances_mean[sorted_idx_rfc], index=x.columns[sorted_idx_rfc]).sort_values(0, ascending=False).iloc[:45]
    
 # gradient boosting
    gb = GradientBoostingClassifier(criterion='friedman_mse',loss='deviance', max_depth=5, n_estimators=30, random_state=66, max_features='auto')
    gb.fit(x, y)
    result_gb = permutation_importance(gb, x,y , n_repeats=10, random_state=66, n_jobs=2)
    sorted_idx_gb = result_gb.importances_mean.argsort()
    importances_gb = pd.DataFrame(result_gb.importances_mean[sorted_idx_gb], index=x.columns[sorted_idx_gb]).sort_values(0, ascending=False).iloc[:45]
    
 # xg boosting
    xg = XGBClassifier(booster='gbtree', max_depth=7,  gamma=0.5, learning_rate=0.01, n_estimators=3, random_state=66)
    xg.fit(x, y)
    result_xg = permutation_importance(xg, x,y , n_repeats=10, random_state=66, n_jobs=2)
    sorted_idx_xg = result_xg.importances_mean.argsort()
    importances_xg = pd.DataFrame(result_xg.importances_mean[sorted_idx_xg], index=x.columns[sorted_idx_xg]).sort_values(0, ascending=False).iloc[:45]
 
    return importances_rf, importances_gb,importances_xg

rfc_list, gb_list, xg_list=clf(x,y,col_list)


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="NanumGothic",
        rc={"axes.unicode_minus":False},
        style='whitegrid')
result = permutation_importance(rfc, x, y, n_repeats=10,
                                random_state=99, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
 
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=x.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


from itertools import combinations
from sklearn.svm import OneClassSVM   #feature set에서 가장 성능이 좋은 조합 확인하는 방법

ocsvm=OneClassSVM( verbose=True, nu=0.00195889, kernel = 'rbf', gamma=0.0009)
 
def oc_model(model, x_train_df, x_test_df, y_test_df):
    model.fit(x_train_df)
    p=model.predict(x_test_df)
    cm = metrics.confusion_matrix(y_test_df, p)
    cm0=cm[0,0]
    cm1=cm[1,1]
    return cm0, cm1
 
 
 
#우선 Univariate Selection은 그룹내 분산이 작고 그룹간 분산이 클 경우 값이 커지는 F-value를 이용하여 변수를 선택한다. 
#각 변수마다 F값을 구해 F값이 큰 변수를 기준으로 변수를 선택하는 방법이다.

from sklearn.feature_selection import SelectKBest, f_classif

selectK = SelectKBest(score_func=f_classif, k=8)
X = selectK.fit_transform(X, y)


#ExtraTreesClassifier와 같은 트리 기반 모델은 Feature Importance 를 제공한다. 이 Feature Importance는 불확실도를 많이 낮출수록 증가하므로 이를 기준으로 변수를 선택할 수 있다.

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier()
etc_model.fit(X, y)

print(etc_model.feature_importances_)
feature_list = pd.concat([pd.Series(X.columns), pd.Series(etc_model.feature_importances_)], axis=1)
feature_list.columns = ['features_name', 'importance']
feature_list.sort_values("importance", ascending =False)[:8]


#마지막으로 RFE (recursive feature elimination)는 Backward 방식중 하나로, 모든 변수를 우선 다 포함시킨 후 반복해서 학습을 진행하면서 중요도가 낮은 변수를 하나씩 제거하는 방식이다.

from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, y)

print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


#Model Selection
# Scikit-learn은 전처리(스케일랑, feature selection, model selection)과 grid search 를 한번에 진행할 수 있도록 파이프라인 기능을 제공한다.

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('classifier', LogisticRegression())])

param_grid = [{'classifier': [SVC()], 
              'classifier__gamma': [0.01, 0.1, 1, 10, 100], 
              'classifier__C': [0.01, 0.1, 1, 10, 100]
              },

               {'classifier': [LogisticRegression()],
               'classifier__penalty': ['l1', 'l2'], 
               'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
               },

              {'classifier': [RandomForestClassifier()],
              'classifier__max_depth': [4, 6], # max_depth: The maximum depth of the tree.
              'classifier__n_estimators': [50, 100], # n_estimators: The number of trees in the forest.
              'classifier__min_samples_split': [50, 100]
              }] # min_samples_split: The minimum number of samples required to split an internal node       

grid = GridSearchCV(pipe, param_grid, scoring = 'roc_auc', cv=5)  
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
