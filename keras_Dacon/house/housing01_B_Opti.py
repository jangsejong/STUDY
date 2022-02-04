from base64 import standard_b64decode
from sqlite3 import paramstyle
import numpy as np
import pandas as pd
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

#이상치 제거

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :", quartile_1)
    print("q2 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score    

# add_metric('NMAE', 'NMAE', NMAE, greater_is_better = False)

path = '../_data/dacon/house/'
train_sets = pd.read_csv(path + 'train.csv', index_col = 0, header =0)
test_sets = pd.read_csv(path + 'test.csv', index_col = 0, header =0)
submit_sets = pd.read_csv(path + 'sample_submission.csv', index_col = 0, header =0)
print(train_sets.shape, test_sets.shape)

print(train_sets.info())
# print(train_sets.description()) #수치형만 나옴
print(train_sets.isnull().sum()) # 결측치 없음

# 중복값 제거
print("제거 전 :", train_sets.shape)
train_sets = train_sets.drop_duplicates()
print("제거 후 :", train_sets.shape)

#이상치 제거
outliers_loc = outliers(train_sets['Garage Yr Blt'])
print("이상치의 위치 :", outliers_loc)
print(train_sets.loc[[255], 'Garage Yr Blt']) #2207
train_sets.drop(train_sets[train_sets['Garage Yr Blt']==2207].index, inplace=True)

print(train_sets.shape, test_sets.shape)


# print(train_sets['Exter Qual'].value_counts())
# print(train_sets['Kitchen Qual'].value_counts())
# print(train_sets['Bsmt Qual'].value_counts()) #po1

# print(test_sets['Exter Qual'].value_counts())
# print(test_sets['Kitchen Qual'].value_counts()) #po1
# print(test_sets['Bsmt Qual'].value_counts())  #po1

# 품질 관련 변수 → 숫자로 매핑
qual_cols = train_sets.dtypes[train_sets.dtypes == np.object].index
#
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

train_sets = label_encoder(train_sets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)
print(train_sets.shape, test_sets.shape)


#분류형 걸럼을 원핫인코딩
train_sets = pd.get_dummies(train_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])


print(train_sets.shape, test_sets.shape)

x = train_sets.drop(['target'], axis=1)
y = train_sets['target']

test_sets = test_sets.values  #  넘파이로 변경

X_train, X_test, y_train, y_test = train_test_split(
  x, y, shuffle=True, random_state=66, train_size=0.8
)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import RobustScaler, MaxAbsScaler
# from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
# from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
# from sklearn.preprocessing import PolynomialFeatures
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = RobustScaler()
# # scaler = MaxAbsScaler()
# # scaler = QuantileTransformer()
# # scaler = PowerTransformer(method='box-cox') #error
# scaler = PowerTransformer(method='yeo-johnson') # default
# # scaler = PolynomialFeatures()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# test_sets = scaler.transform(test_sets)


######BayesianOptimization
params = {'max_depth': (3, 7),
          'learning_rate': (0.01, 0.2),
          'n_estimators': (5000, 10000),
          #'gamma':(0,100),
          'min_child_weight': (0,3),
          'subsample': (0.5, 1),
          'colsample_bytree':(0.2, 1),
          'reg_lambda': (0.001, 10), 
          # 'reg_alpha':(0.01, 50)
          }

def xg_def(max_depth,learning_rate,n_estimators,min_child_weight,subsample,colsample_bytree,reg_lambda):
      xg_model = XGBRegressor(
        max_depth = int(max_depth),
        learning_rate = learning_rate,
        n_estimators = int(n_estimators),
        min_child_weight = min_child_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        reg_lambda = reg_lambda
      )
      
      xg_model.fit(X_train, y_train,eval_set=([X_test, y_test]) ,eval_metric='mae', verbose=1, early_stopping_rounds=50)
      y_predict = xg_model.predict(X_test)
      
      nmae = NMAE(y_test, y_predict)
      return nmae


from pickletools import optimize
from bayes_opt import BayesianOptimization

bo = BayesianOptimization(f= xg_def, pbounds=params, random_state=66, verbose=2)

bo.maximize(init_points=10, n_iter=200)

print(bo.res)
print("============파라미터 튜닝 결과================")
print(bo.max)

target_list = []
for result in bo.res:
    target = result['target']
    target_list.append(target)
    
min_dict = bo.res[np.argmin(np.array(target_list))]
print(min_dict)

# def black_box_funtion(x, y):
#     return - x **2 - (y - 1) ** 2 + 1

# pbounds = {'x' : (2, 4), 'y' : (-3, 3)}

# optimizer = BayesianOptimization(
#     f = black_box_funtion,
#     pbounds=pbounds,
#     random_state=66
# )

# optimizer.maximize(
#     init_points = 2,
#     n_iter=15
# )