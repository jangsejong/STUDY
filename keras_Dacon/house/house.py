SEED = 66


import pandas as pd
train_data=pd.read_csv('./././_data/dacon/house/train.csv')
test_data=pd.read_csv('./././_data/dacon/house/test.csv')

print('train_data shape : ', train_data.shape)
print('test_data shape : ', test_data.shape)
# train_data shape :  (1350, 15)
# test_data shape :  (1350, 14)


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

train_data=cut_outlier(train_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])
test_data=cut_outlier(test_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])


# 중복값 제거
print("제거 전 :", train_data.shape)
train_data = train_data.drop_duplicates()
print("제거 후 :", train_data.shape)



# train[train['Garage Yr Blt']> 2050] # 254
# train.loc[254, 'Garage Yr Blt'] = 2007
train_data.drop(train_data[train_data['Garage Yr Blt']==2207].index, inplace=True)


#데이터 변수 순서 정리해 줌
cols1=['id', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt',  'Overall Qual', 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual', 'Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'Full Bath', 'Garage Cars', 'target']
train_data=train_data[cols1]
cols2=['id', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt',  'Overall Qual', 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual', 'Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'Full Bath', 'Garage Cars']
test_data=test_data[cols2]
train_data.head(3)

#년도와 관련된 변수 2022-변수로 바꿔주기
train_data['Year Built']=2022-train_data['Year Built']
train_data['Year Remod/Add']=2022-train_data['Year Remod/Add']
train_data['Garage Yr Blt']=2022-train_data['Garage Yr Blt']
test_data['Year Built']=2022-test_data['Year Built']
test_data['Year Remod/Add']=2022-test_data['Year Remod/Add']
test_data['Garage Yr Blt']=2022-test_data['Garage Yr Blt']

#이상치 제거
def cut_outlier(df2, columns):
    df=df2.copy()
    for column in columns:
        q1=df[column].quantile(.255)
        q3=df[column].quantile(.745)
        iqr=q3-q1
        low=q1-1.5*iqr
        high=q3+1.5*iqr
        df.loc[df[column]<low, column]=low
        df.loc[df[column]>high, column]=high
    return df

train_data_2=cut_outlier(train_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])
test_data_2=cut_outlier(test_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])

import seaborn as sns
import matplotlib.pyplot as plt

data_corr=train_data_2.corr()
#히트맵
plt.figure(figsize=(10,10))
sns.set(font_scale=0.8)
sns.heatmap(data_corr, annot=True, cbar=False)
# plt.show()

#Target 변수와 상관관계가 높은 순으로 출력
corr_order=train_data_2.corr().loc[:'Garage Cars', 'target'].abs().sort_values(ascending=False) #abs() : 절댓값 붙이기 #sort_values(ascending=False) : 오름차순 정렬
corr_order

# 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual' 변수는 Overall Qual로 보면 되기 때문에 삭제
train_data_3=train_data_2.drop(['Exter Qual', 'Bsmt Qual', 'Kitchen Qual'], axis=1, inplace=False)
test_data_3=test_data_2.drop(['Exter Qual', 'Bsmt Qual', 'Kitchen Qual'], axis=1, inplace=False)
train_data_3.head(3)

X_data=train_data_3[['Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Overall Qual', 'Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'Full Bath', 'Garage Cars']]
Y_data=train_data_3[['target']]            

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_data, Y_data, test_size=0.2, random_state=SEED)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor #랜덤포레스트회귀, GBM회귀
from sklearn.tree import DecisionTreeRegressor #의사결정회귀
from xgboost import XGBRegressor #XGB회귀
from lightgbm import LGBMRegressor #LGB회귀



# rf_reg=RandomForestRegressor(random_state=SEED, n_estimators=15000, criterion="mae")
# gb_reg=GradientBoostingRegressor(random_state=SEED, n_estimators=10000)
# dt_reg=DecisionTreeRegressor(random_state=SEED, max_depth=4)
xgb_reg=XGBRegressor(n_estimators=15000)
# lgb_reg=LGBMRegressor(n_estimators=1000)

# dt_reg.fit(X_train,y_train)
# rf_reg.fit(X_train,y_train)
# gb_reg.fit(X_train,y_train)
xgb_reg.fit(X_train,y_train)
# lgb_reg.fit(X_train,y_train)

# y_preds_dt=dt_reg.predict(X_test)
# y_preds_rf=rf_reg.predict(X_test)
# y_preds_gb=gb_reg.predict(X_test)
y_preds_xgb=xgb_reg.predict(X_test)
# y_preds_lgb=lgb_reg.predict(X_test)

# y_test.shape #정답

# y_preds_dt.shape #예측값

# result_dt=pd.DataFrame(y_preds_dt, index=y_test.index).rename(columns={0: 'prediction_dt'})
# result_rf=pd.DataFrame(y_preds_rf, index=y_test.index).rename(columns={0: 'prediction_rf'})
# result_gb=pd.DataFrame(y_preds_gb, index=y_test.index).rename(columns={0: 'prediction_gb'})
result_xgb=pd.DataFrame(y_preds_xgb, index=y_test.index).rename(columns={0: 'prediction_xgb'})
# result_lgb=pd.DataFrame(y_preds_lgb, index=y_test.index).rename(columns={0: 'prediction_lgb'})

result=pd.concat([y_test, result_xgb], axis=1)
result

import numpy as np

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

# result['score_dt']=NMAE(result['target'], result['prediction_dt'])
# result['score_rf']=NMAE(result['target'], result['prediction_rf'])
# result['score_gb']=NMAE(result['target'], result['prediction_gb'])
result['score_xgb']=NMAE(result['target'], result['prediction_xgb'])
# result['score_lgb']=NMAE(result['target'], result['prediction_lgb'])

# print('decision tree 의 score :', np.mean(result['score_dt']))
# print('random forest 의 score :', np.mean(result['score_rf']))
# print('GBM 의 score :', np.mean(result['score_gb']))
print('XGB 의 score :', np.mean(result['score_xgb']))
# print('LGB 의 score :', np.mean(result['score_lgb']))

model = XGBRegressor(max_depth = 4,
        learning_rate = 0.035,
        n_estimators = 6000,
        min_child_weight = 2,
        subsample =  0.87,
        colsample_bytree = 0.32,
        reg_lambda = 2.51
    
)
model.fit(X_data, Y_data) # train data로 학습
test_data_3=test_data_3.drop(['id'], axis=1, inplace=False) #X_data의 columns에 id변수가 없으므로 test_data_3에도 id변수를 없애준다
y_preds=model.predict(test_data_3)
'''
'target': 0.09295295826990345, 'params': {'colsample_bytree': 0.32343006250502065, 
'learning_rate': 0.03540291653301253, 'max_depth': 4.450741864707551, 
'min_child_weight': 2.0373266263368945, 'n_estimators': 5972.2502866693385, 
'reg_lambda': 2.5128526244864484, 'subsample': 0.8792081957678033}}
'''
result_sub=pd.DataFrame(y_preds, index=test_data_3.index).rename(columns={0: 'target'})
id_sub=test_data[['id']]

# print(result_sub.head(3))

# print(id_sub.head(3))

submission=pd.concat([id_sub, result_sub], axis=1)

submission.to_csv('./././_data/dacon/house/house_0204_001.csv', index=False)



