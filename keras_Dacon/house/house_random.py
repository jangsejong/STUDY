SEED = 42

import pandas as pd
import os
import os.path as osp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
path = '../_data/dacon/house'
train = pd.read_csv(path + 'train.csv', index_col = 0, header =0)
test = pd.read_csv(path + 'test.csv', index_col = 0, header =0)

print('train_data shape : ', train.shape)
print('test_data shape : ', test.shape)
# train_data shape :  (1350, 15)
# test_data shape :  (1350, 14)





# train[train['Garage Yr Blt']> 2050] # 254
train.loc[254, 'Garage Yr Blt'] = 2007

# 품질 관련 변수 → 숫자로 매핑
qual_cols = train.dtypes[train.dtypes == np.object].index
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

train = label_encoder(train, qual_cols)
test = label_encoder(test, qual_cols)
train.head()


def feature_eng(data_):
  data = data_.copy()
  data['Year Gap Remod'] = data['Year Remod/Add'] - data['Year Built']
  data['Car Area'] = data['Garage Area']/data['Garage Cars']
  data['2nd flr SF'] = data['Gr Liv Area'] - data['1st Flr SF']
  data['2nd flr'] = data['2nd flr SF'].apply(lambda x : 1 if x > 0 else 0)
  data['Total SF'] = data[['Gr Liv Area',"Garage Area", "Total Bsmt SF"]].sum(axis=1)
  data['Sum Qual'] = data[["Exter Qual", "Kitchen Qual", "Overall Qual"]].sum(axis=1)
  data['Garage InOut'] = data.apply(lambda x : 1 if x['Gr Liv Area'] != x['1st Flr SF'] else 0, axis=1)
  return data

train = feature_eng(train)
test = feature_eng(test)

#이상치 제거
def cut_outlier(df2, columns):
    df=df2.copy()
    for column in columns:
        q1=df[column].quantile(.25)
        q3=df[column].quantile(.75)
        iqr=q3-q1
        low=q1-1.5*iqr
        high=q3+1.5*iqr
        df.loc[df[column]<low, column]=low
        df.loc[df[column]>high, column]=high
    return df

train=cut_outlier(train, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])
test=cut_outlier(test, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])

# 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual' 변수는 Overall Qual로 보면 되기 때문에 삭제
train=train.drop(['Exter Qual', 'Bsmt Qual', 'Kitchen Qual'], axis=1, inplace=False)
test=test.drop(['Exter Qual', 'Bsmt Qual', 'Kitchen Qual'], axis=1, inplace=False)

 


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

# 평가 기준 정의
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

nmae_score = make_scorer(NMAE, greater_is_better=False)
kf = KFold(n_splits = 10, random_state = SEED, shuffle = True)

X = train.drop(['target'], axis = 1)
y = np.log1p(train.target)

target = test[X.columns]

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# RandomForestRegressor
rf_pred = np.zeros(target.shape[0])
rf_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rf = RandomForestRegressor(random_state=SEED, n_estimators=15000, criterion="mae")
    rf.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rf.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rf_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = rf.predict(target) / 10
    rf_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rf_val)} & std = {np.std(rf_val)}')

# # Catboost
# cb_pred = np.zeros(target.shape[0])
# cb_val = []
# for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
#     print(f'{n + 1} FOLD Training.....')
#     tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
#     val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
#     tr_data = Pool(data = tr_x, label = tr_y)
#     val_data = Pool(data = val_x, label = val_y)
    
#     cb = CatBoostRegressor(depth = 4, random_state = 42, loss_function = 'MAE', n_estimators = 3000, learning_rate = 0.03, verbose = 0)
#     cb.fit(tr_data, eval_set = val_data, early_stopping_rounds = 750, verbose = 1000)
    
#     val_pred = np.expm1(cb.predict(val_x))
#     val_nmae = NMAE(val_y, val_pred)
#     cb_val.append(val_nmae)
#     print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
#     target_data = Pool(data = target, label = None)
#     fold_pred = cb.predict(target) / 10
#     cb_pred += fold_pred
# print(f'10FOLD Mean of NMAE = {np.mean(cb_val)} & std = {np.std(cb_val)}')


# 검증 성능 확인하기
val_list = [ rf_val]
for val in val_list :
  print("{:.8f}".format(np.mean(val))) 
  
  
# submission=pd.concat([id_sub, result_sub], axis=1)

# submission.to_csv('../_data/dacon/house/house_0203_04.csv', index=False)

# submission 파일에 입력
sub = pd.read_csv(osp.join(path, 'sample_submission.csv'))
sub['target'] = np.expm1(rf_pred)
sub['target']

# csv 파일로 내보내기
sub.to_csv(osp.join(path, 'house_0203_04.csv'), index=False) 


'''
seed=42
0.09666806


'''
