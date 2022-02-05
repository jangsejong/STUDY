import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

path = 'D:\\Study\\_data\\dacon\\house\\'
train = pd.read_csv(path + 'train.csv', index_col = 0, header =0)
test = pd.read_csv(path + 'test.csv', index_col = 0, header =0)
submission = pd.read_csv(path + 'sample_submission.csv')

train = train.iloc[:, 1:]
test = test.iloc[:, 1:]


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

# 중복값 제거
print("제거 전 :", train.shape)
train = train.drop_duplicates()
print("제거 후 :", train.shape)

cat_cols = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual']
train.drop( train[train['Garage Yr Blt']==2207].index, inplace=True )


for c in cat_cols :
    ord_df = train.groupby(c).target.median().reset_index(name = f'ord_{c}')
    train = pd.merge(train, ord_df, how = 'left')
    test = pd.merge(test, ord_df, how = 'left')

train.drop(cat_cols, axis = 1, inplace = True)
test.drop(cat_cols, axis = 1, inplace = True)

print(f'로그 변환 전 타겟 왜도 = {train.target.skew()} / 로그 변환 후 타겟 왜도 = {np.log1p(train.target).skew()}')

X = train.drop('target', axis = 1)
y = np.log1p(train.target)

target = test[X.columns]

target.fillna(target.mean(), inplace = True)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

def NMAE(true, pred) -> float:
    mae = np.mean(np.abs(true - pred))
    score = mae / np.mean(np.abs(true))
    return score

nmae_score = make_scorer(NMAE, greater_is_better=False)

random_state = 77

kf = KFold(n_splits = 10, random_state = random_state, shuffle = True)

n_estimators = 10000


#RandomForest
rf_pred = np.zeros(target.shape[0])
rf_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rf = RandomForestRegressor(random_state = random_state, criterion = 'mae', n_estimators = n_estimators)
    rf.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rf.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rf_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = rf.predict(target) / 10
    rf_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rf_val)} & std = {np.std(rf_val)}')

#GradientBoosting
gbr_pred = np.zeros(target.shape[0])
gbr_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    gbr = GradientBoostingRegressor(random_state = random_state, max_depth = 4, learning_rate = 0.04, n_estimators = n_estimators)
    gbr.fit(tr_x, tr_y)
    
    val_pred = np.expm1(gbr.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    gbr_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = gbr.predict(target) / 10
    gbr_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(gbr_val)} & std = {np.std(gbr_val)}')

#CatBoost
cb_pred = np.zeros(target.shape[0])
cb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    tr_data = Pool(data = tr_x, label = tr_y)
    val_data = Pool(data = val_x, label = val_y)
    
    cb = CatBoostRegressor(depth = 4, random_state = random_state, loss_function = 'MAE', n_estimators = n_estimators, learning_rate = 0.03, verbose = 0)
    cb.fit(tr_data, eval_set = val_data, early_stopping_rounds = 3000, verbose = 100)
    
    val_pred = np.expm1(cb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    cb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = cb.predict(target) / 10
    cb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(cb_val)} & std = {np.std(cb_val)}')

#NGBoost
ngb_pred = np.zeros(target.shape[0])
ngb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    ngb = NGBRegressor(random_state = random_state, n_estimators = n_estimators, verbose = 0, learning_rate = 0.04)
    ngb.fit(tr_x, tr_y, val_x, val_y, early_stopping_rounds = 3000)
    
    val_pred = np.expm1(ngb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    ngb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = ngb.predict(target) / 10
    ngb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(ngb_val)} & std = {np.std(ngb_val)}')
from xgboost import XGBRegressor

#XGBoost
xgb_pred = np.zeros(target.shape[0])
xgb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    xgb = XGBRegressor(random_state = random_state, criterion = 'mae', n_estimators = n_estimators)
    xgb.fit(tr_x, tr_y)
    
    val_pred = np.expm1(xgb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    xgb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = xgb.predict(target) / 10
    xgb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(xgb_val)} & std = {np.std(xgb_val)}')





(ngb_pred + cb_pred + rf_pred + gbr_pred + xgb_pred) / 5

submission['target'] = np.expm1((ngb_pred + cb_pred + rf_pred + gbr_pred + xgb_pred) / 5)

submission.to_csv(path +'1more_01.csv', index = False)




