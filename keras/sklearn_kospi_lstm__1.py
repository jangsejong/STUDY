import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.core.indexes.datetimes import date_range
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
import pickle
import platform
import os
import warnings
warnings.filterwarnings('ignore')

#데이터 로드

path = 'D:\\Study\\개인프로젝트\\데이터자료\\csv\\'

# covid_19 = pd.read_csv(path +"코로나바이러스감염증-19_확진환자_발생현황_220107.csv", thousands=",", encoding='cp949')
kospi = pd.read_csv(path +"코스피지수(202001~202112).csv", thousands=",", encoding='cp949')

# print(covid_19.head())
# print(kospi.head())

#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.
kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)

# print(covid_19.info())
print(kospi.info())


# 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
# covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
kospi['new_Date'] = pd.to_datetime(kospi['일자'])

# covid_19.drop('일자', axis = 1, inplace=True)
# covid_19.set_index('new_Date', inplace=True)

kospi.drop('일자', axis = 1, inplace=True)
kospi.set_index('new_Date', inplace=True)
# print(x1.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
# x1 = covid_19.drop(columns=[], axis=1) 
x2 = kospi.drop(columns =['대비','등락률(%)', '배당수익률(%)', '주가이익비율', '주가자산비율', '거래량(천주)', '상장시가총액(백만원)', '거래대금(백만원)'], axis=1) 


#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 삭제하여 준다.
import holidays
kr_holidays = holidays.KR(years=[2020,2021,2022])
# print(kr_holidays)

x22 = np.array(x2)

print(x2.info())

def split_xy3(dataset, time_steps, y_column):                 
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps                    # 0 + 5 = 5
        y_end_number = x_end_number + y_column      # 5 + 2 = 7
        
        if y_end_number > len(dataset):
            break
             
        tmp_x = dataset[i:x_end_number, 1:]                #0:5,  1:
        tmp_y = dataset[x_end_number:y_end_number, 0]    #5:7, 0
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x2_ko, y2_ko = split_xy3(x22, 5, 2)

from sklearn.model_selection import train_test_split

x2_train, x2_test, y2_train, y2_test = train_test_split(x2_ko, y2_ko ,train_size=0.8, random_state=66)

# print(x1_train.shape) #(20, 5, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(LSTM(16, activation='tanh', input_shape=(5, 3)))
model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 20 , mode = 'auto', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1220_lastcost7.hdf6')

model.fit(x2_train, y2_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])#,mcp]) 

# model.save_weights("./_save/keras999_1_save_weights.h5")

#model = load_model("./_ModelCheckPoint/ss_ki_1220_lastcost5.hdf5")
#4. 평가, 예측
loss = model.evaluate ([x2_test], [y2_test], batch_size=1)
print('loss :', loss) #loss :
y2_pred = model.predict([x2_test])
print('코스피예상지수 : ', y2_pred[-1])
print(y2_pred[:5])

'''
코스피예상지수 :  [2706.0867]
코스피예상지수 :  [2705.1616]
코스피예상지수 :  [2704.216]
코스피예상지수 :  [2700.6172]
코스피예상지수 :  [2706.017]
코스피예상지수 :  [2705.435]
코스피예상지수 :  [2703.2097]
코스피예상지수 :  [2705.1052]
코스피예상지수 :  [2706.6223]
코스피예상지수 :  [2705.0889]

10번 결과값 평균 : 2705
'''

