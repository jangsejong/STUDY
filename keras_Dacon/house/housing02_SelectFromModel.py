
from random import random
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor


########################################################## [[ 아웃라이어 함수 정의 ]] ##########################################################
def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quantile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quantile_3)
    iqr = quantile_3 - quantile_1
    print("iqr : ", iqr)
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |        #  이 줄 또는( | )
                    (data_out<lower_bound))         #  아랫줄일 경우 반환
################################################################ [[ 평가 기준 정의 ]] ################################################################
# NMAE 함수 정의
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
################################################################ (함수라 맨위에) ################################################################


path = '../_data/dacon/house/'


datasets = pd.read_csv(path+'train.csv', index_col=0, header=0)
test_sets = pd.read_csv(path+ 'test.csv',index_col=0, header=0)
sumit_sets = pd.read_csv(path+ 'sample_submission.csv',index_col=0, header=0)
print(datasets)
print(datasets.info())

print(datasets.describe())      # 컬럼 확인
print(datasets.isnull().sum())  # 결측치 확인


########################################################## 중복값 처리 ##########################################################
print("중복값 제거전 : ", datasets.shape) # 중복값 제거전 :  (1350, 14)
datasets = datasets.drop_duplicates()
print("중복값 제거후 : ", datasets.shape) # 중복값 제거후 :  (1349, 14)


########################################################## 이상치 확인 처리 ##########################################################
# 상단에 함수 존재
outliers_loc = outliers( datasets['Garage Yr Blt'] )
print( outliers_loc )
print("이상치의 위치 : ", outliers_loc)
# print( datasets.loc[[254], 'Garage Yr Blt'] ) # 행, 열        #정상
print( datasets.loc[[255], 'Garage Yr Blt'] ) # 행, 열          #이상
# 2207년의 행 하나 드랍
datasets.drop( datasets[datasets['Garage Yr Blt']==2207].index, inplace=True )
print(datasets.shape)#확인 (1348, 14)       (=> duplicate와 2207년이 제거됨)
print(datasets['Exter Qual'].value_counts() )
print(datasets['Bsmt Qual'].value_counts() )
# print(test_sets['Exter Qual'].value_counts() )
# print(test_sets['Kitchen Qual'].value_counts() )
print(test_sets['Bsmt Qual'].value_counts() )
########################################################## 라벨 인코더 대신 수동 인코더 ##########################################################
# 품질 관련 변수 → 숫자로 매핑
qual_cols = datasets.dtypes[datasets.dtypes == np.object].index
def label_encoder(df_, qual_cols):
    df = df_.copy()
    mapping={
        'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
    }
    for col in qual_cols :
        df[col] = df[col].map(mapping)
    return df

datasets = label_encoder(datasets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)
print(datasets.shape) # (1350, 14)
print(test_sets.shape) # (1350, 13)

########################################################## 분류형 컬럼을 One Hot Encoding ##########################################################
datasets = pd.get_dummies(datasets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])


########################################################## One Hot Encoding 확인용 ##########################################################
print(datasets.shape)       #(1350, 24)    =>  (1350, 23)
print(test_sets.shape)      #(1350, 24)    =>  (1350, 22)
########################################################## xy 분리 ##########################################################
x= datasets.drop(['target'], axis=1)
y = datasets['target']

test_sets = test_sets.values        # 넘파이로 변경

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_train.shape, y_train.shape)#(1080, 22) (1080,)
print(x_test.shape, y_test.shape)#(270, 22) (270,)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer(method='box-cox')     # error
scaler = PowerTransformer(method='yeo-johnson') # default

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

# ########################################################## 베이지안 옵티마이제이션 사용 ##########################################################
# parms = {'max_depth':(3,7),
#          'learning_rate':(0.01, 0.2),
#          'n_estimators':(5000, 10000),
#          'min_child_weight':(0,3),
#          'subsample':(0.5, 1),
#          'colsample_bytree':(0.2,1),
#          'reg_lambda':(0.001,10),
# }

# def xg_def(max_depth, learning_rate, n_estimators,
#            min_child_weight, subsample, colsample_bytree,
#            reg_lambda
#           ):
#         xg_model = XGBRegressor(
#             max_depth = int(max_depth),
#             learning_rate = learning_rate,
#             n_estimators = int(n_estimators),
#             min_child_weight = min_child_weight,
#             subsample = subsample,
#             colsample_bytree = colsample_bytree,
#             reg_lambda = reg_lambda
#         )
#         xg_model.fit(x_train, y_train,
#                     eval_set= [(x_test, y_test)],
#                     eval_metric='mae',
#                     verbose=1,
#                     early_stopping_rounds=50
#                     )
#         y_predict = xg_model.predict(x_test)
        
#         nmae = NMAE(y_test, y_predict)
#         return nmae
    
    

# ########################################################## 베이지안 옵티마이제이션 정의 ##########################################################
# bo = BayesianOptimization(f=xg_def,
#                           pbounds=parms,
#                           random_state=66,
#                           verbose=2
#                          )

# bo.maximize(init_points=10, n_iter=200) 


# print("=========================== bo.res ===========================")
# print((bo.res))
# print("=========================== 파라미터 튜닝 결과 ===========================")
# print((bo.max))

# target_list = []
# for result in bo.res:
#     target = result['target']
#     target_list.append(target)
    
# min_dict = bo.res[np.argmin(np.array(target_list))]
# print(min_dict)


# target = 0.09295295826990345
colsample_bytree= 0.323
learning_rate= 0.035
max_depth= 5
min_child_weight= 2.037
n_estimators=5972
reg_lambda=2.512
subsample= 0.879



###################################### SelectFromModel

model = XGBRegressor(n_jobs= -1,colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                     n_estimators=n_estimators, reg_lambda=reg_lambda, subsample=subsample )


model.fit(x_train, y_train,
                    eval_set= [(x_test, y_test)],
                    eval_metric='mae',
                    verbose=1,
                    early_stopping_rounds=50
                    )
# print(datasets)
thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs= -1,colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                     n_estimators=n_estimators, reg_lambda=reg_lambda, subsample=subsample )
    selection_model.fit(select_x_train, y_train, eval_set= [(select_x_test, y_test)],
                    eval_metric='mae',
                    verbose=0,
                    early_stopping_rounds=100
                    )
    
    y_predict = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))