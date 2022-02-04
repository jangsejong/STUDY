
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

from sklearn.metrics import make_scorer

nmae_score = make_scorer(NMAE, greater_is_better=False)

################################################################ (함수라 맨위에) ################################################################


path = '../_data/dacon/house/'


datasets = pd.read_csv(path+'train.csv', index_col=0, header=0)
test_sets = pd.read_csv(path+ 'test.csv',index_col=0, header=0)
submit_sets = pd.read_csv(path+ 'sample_submission.csv',index_col=0, header=0)
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

### 로그변환


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

import autokeras as ak

ak_model = ak.StructuredDataRegressor(overwrite=True, max_trials=5, loss='mean_absolute_error')

start = time.time()
ak_model.fit(x_train, y_train, epochs=2)
end =  time.time() - start


model = ak_model.export_model()
y_predict = model.predict(x_test)
y_predict = y_predict.reshape(270,)

results = model.evaluate(x_test, y_test)
print("loss :", np.round(results,6))

nmae = NMAE(np.expm1(y_test), np.expm1(y_predict))
print("NMAE :", np.round(nmae,6))

# score
y_submit = model.prtdict(test_sets)

y_submit = np.expm1(y_submit)
submit_sets.target = y_submit


path_csv = '../_data/dacon/house/'
now1 = datetime.now()

now_data = now1.starftime("%m%d_%H%M") #연도와초를 빼본다.

submit_sets.to_csv(path_csv + now_data + '_'+ str(round(nmae,4)) + '.csv')


colsample_bytree= 0.323
learning_rate= 0.035
max_depth= 5
min_child_weight= 2.037
n_estimators=5972
reg_lambda=2.512
subsample= 0.879


with open(path_csv + now_data + '_' + str(round(nmae,4)) + 'submit.txt','a') as file:
    file.write("\n=========================")
    file.write("저장시간 :"+ now_data + '\n')
    file.write("scaler :" + str(scaler) + '\n')
    file.write("colsample_bytree :" + str('colsample_bytree') + '\n')
    file.write("learning_rate :" + str('learning_rate') + '\n')
    file.write("max_depth :" + str('max_depth') + '\n')
    file.write("min_child_weight :" + str('min_child_weight') + '\n')
    file.write("n_estimators :" + str('n_estimators') + '\n')
    file.write("reg_lambda :" + str('reg_lambda') + '\n')
    file.write("subsample :" + str('subsample') + '\n')
    file.write("걸린시간 :" + str(round(end,4)) + '\n')
    file.write("evaluate :" + str(np.round(results,6)) + '\n')
    file.write("nmae :" + str(round(nmae,6)) + '\n')
    
