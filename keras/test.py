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
import platform
import os
import warnings
warnings.filterwarnings('ignore')

#데이터 로드

path = 'D:\\Study\\개인프로젝트\\데이터자료\\csv\\'

covid_19 = pd.read_csv(path +"코로나바이러스감염증-19_확진환자_발생현황_220110.csv", thousands=",", encoding='cp949')
kospi = pd.read_csv(path +"코스피지수(202001~202112)_결측치.csv", thousands=",", encoding='cp949')




#데이터 전처리


# kospi= kospi.fillna(method = 'pad') #결측치 바로 이전 값으로 채우기
# kospi= kospi.fillna(method = 'bfill') #결측치 바로 이후 값으로 채우기
# # 컬럼의 평균으로 대체하기
# kospi['현재지수'] = kospi['현재지수'].fillna(kospi['현재지수'].mean()) 
# kospi['주가이익비율'] = kospi['주가이익비율'].fillna(kospi['주가이익비율'].mean()) 
# kospi['시가지수'] = kospi['시가지수'].fillna(kospi['시가지수'].mean()) 
# kospi['저가지수'] = kospi['저가지수'].fillna(kospi['저가지수'].mean()) 
# kospi = kospi.fillna()  # 0으로 대체하기

#covid_19.csv 에서 결측치 칼럼이 확인되어 삭제해주었다.

# covid_19 = covid_19.drop(covid_19.index[range(0,1)] ,axis=0)
covid_19 = covid_19.drop(covid_19.columns[range(5,8)], axis=1)
# covid_19 = covid_19.replace("-","0")


print(covid_19.head())
print(kospi.head())
#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.
kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)




# 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
kospi['new_Date'] = pd.to_datetime(kospi['일자'])

covid_19.drop('일자', axis = 1, inplace=True)
covid_19.set_index('new_Date', inplace=True)

kospi.drop('일자', axis = 1, inplace=True)
kospi.set_index('new_Date', inplace=True)
# print(x1.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
x1 = covid_19.drop(columns=[], axis=1) 
x2 = kospi.drop(columns =['대비','등락률(%)', '고가지수', '주가자산비율', '거래량(천주)', '상장시가총액(백만원)', '거래대금(백만원)','배당수익률(%)'], axis=1) 

for col1 in x1.columns:
    n_nan1 = x1[col1].isnull().sum()
    if n_nan1>0:
      msg1 = '{:^20}에서 결측치 개수 : {}개'.format(col1,n_nan1)
      print(msg1)
    else:
          print('결측치가 없습니다.')    
for col2 in x2.columns:
    n_nan2 = x2[col2].isnull().sum()
    if n_nan2>0:
      msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
      print(msg2)
    else:
        print('결측치가 없습니다.')  

# print(x1.head())
# print(x2.head())
# print(x1.shape)
# print(x2.shape)

'''
#주식장이 열리지 않는 주말에 해당하는 행을 삭제하여 준다.
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SAT'),axis=0)
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SUN'),axis=0)


#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 확인해보았다.

import holidays
import pandas as pd


kr_holidays = holidays.KR(years=[2020,2021]) 


'''
'''
{datetime.date(2020, 1, 1): "New Year's Day", datetime.date(2020, 1, 24): "The day preceding of Lunar New Year's Day", 
datetime.date(2020, 1, 25): "Lunar New Year's Day", datetime.date(2020, 1, 26): "The second day of Lunar New Year's Day", 
datetime.date(2020, 1, 27): "Alternative holiday of Lunar New Year's Day", datetime.date(2020, 3, 1): 'Independence Movement Day', 
datetime.date(2020, 4, 30): 'Birthday of the Buddha', datetime.date(2020, 5, 5): "Children's Day", datetime.date(2020, 5, 1): 'Labour Day', 
datetime.date(2020, 6, 6): 'Memorial Day', datetime.date(2020, 8, 15): 'Liberation Day', datetime.date(2020, 9, 30): 'The day preceding of Chuseok', 
datetime.date(2020, 10, 1): 'Chuseok', datetime.date(2020, 10, 2): 'The second day of Chuseok', datetime.date(2020, 10, 3): 'National Foundation Day', 
datetime.date(2020, 10, 9): 'Hangeul Day', datetime.date(2020, 12, 25): 'Christmas Day', datetime.date(2020, 8, 17): 'Alternative public holiday'}
{datetime.date(2021, 1, 1): "New Year's Day", datetime.date(2021, 2, 11): "The day preceding of Lunar New Year's Day", 
datetime.date(2021, 2, 12): "Lunar New Year's Day", datetime.date(2021, 2, 13): 
"The second day of Lunar New Year's Day", datetime.date(2021, 3, 1): 'Independence Movement Day', datetime.date(2021, 5, 19): 'Birthday of the Buddha', 
datetime.date(2021, 5, 5): "Children's Day", datetime.date(2021, 5, 1): 'Labour Day', datetime.date(2021, 6, 6): 'Memorial Day', 
datetime.date(2021, 8, 15): 'Liberation Day', datetime.date(2021, 8, 16): 'Alternative holiday of Liberation Day', 
datetime.date(2021, 9, 20): 'The day preceding of Chuseok', datetime.date(2021, 9, 21): 'Chuseok', datetime.date(2021, 9, 22): 'The second day of Chuseok', 
datetime.date(2021, 10, 3): 'National Foundation Day', datetime.date(2021, 10, 4): 'Alternative holiday of National Foundation Day', 
datetime.date(2021, 10, 9): 'Hangeul Day', datetime.date(2021, 10, 11): 'Alternative holiday of Hangeul Day', datetime.date(2021, 12, 25): 'Christmas Day'}
'''
'''



date_list=[]

for date, occasion in kr_holidays.items():
    date = (f'{date}')
    date_list.append(date)
    date_list.sort()
  
# print(date_list)
'''
['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-03-01', '2020-04-30', '2020-05-01', '2020-05-05', 
'2020-06-06', '2020-08-15', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09', '2020-12-25', 
'2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01', '2021-05-01', '2021-05-05', '2021-05-19', '2021-06-06', 
'2021-08-15', '2021-08-16', '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04', '2021-10-09', '2021-10-11', '2021-12-25']
'''


date_list.remove('2020-01-01')
date_list.remove('2020-01-25')
date_list.remove('2020-01-26')
date_list.remove('2020-03-01')
date_list.remove('2020-06-06')
date_list.remove('2020-08-15')
date_list.remove('2020-10-03')
# date_list.remove('2020-10-09')
# date_list.remove('2020-12-25')
date_list.remove('2021-02-13')
date_list.remove('2021-05-01')
date_list.remove('2021-06-06')
date_list.remove('2021-08-15')
date_list.remove('2021-10-03')
date_list.remove('2021-10-09')
date_list.remove('2021-12-25')

# print(date_list)
'''
['2020-01-24', '2020-01-27', '2020-04-30', '2020-05-01', '2020-05-05', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02',
'2021-01-01', '2021-02-11', '2021-02-12', '2021-03-01', '2021-05-05', '2021-05-19', '2021-08-16', '2021-09-20', '2021-09-21',
'2021-09-22', '2021-10-04', '2021-10-11']
'''
x1.drop(date_list[0:], axis=0, inplace=True)

x1.drop("2020-04-15", axis=0, inplace=True)  #제21대 국회의원선거
x1.drop("2020-12-31", axis=0, inplace=True)  #한국 폐장일
x1.drop("2021-12-31", axis=0, inplace=True)  #한국 폐장일


# x1.drop("2020-10-09", axis=0, inplace=True)
'''



# print(x1.info)
# print(x2.info)
print(x1.head)
print(x2.head)
print(x1.shape)
print(x2.shape)



# from pytimekr import pytimekr
 
# x1 = np.array(x1)
# x2 = np.array(x2)
# print(x1.head, x2.head)

# x1.plot(x='일자', y='계(명)')
# x2.plot(x='일자', y='시가지수')

# plt.figure(figsize = (9, 5))
# plt.grid()
# plt.legend(loc = 'upper right')
# plt.show()

# plt.figure(figsize=(16, 9))
# sns.lineplot(y=x1['계(명)'], x=x1.index)
# plt.xlabel('new_Date')
# plt.ylabel('계(명)')
# plt.show()   

# print(x1.shape, x2.shape) #(484, 4) (484, 4)

# #slice 사용하여 칼럼 위치 변경 (일별환자수보다 누적 환자수를 타켓값으로 줄때 값이 좋게 나왔다)

# col1=x1.columns[-1:].to_list()
# col2=x1.columns[:1].to_list()
# new_col=col1+col2
# x1=x1[new_col]
# print(x1.head)
X11 = x1.drop(['계(명)'], axis=1)
X22 = x2.drop(['현재지수'], axis=1)
X11=np.asarray(X11).astype(np.int)
X22=np.asarray(X22).astype(np.int)
print(X11)
print(X22)

x11 = np.array(x1)
x22 = np.array(x2)

print(x1.info())
print(x2.info())
x11=np.asarray(x1).astype(np.int)
x22=np.asarray(x2).astype(np.int)

# scaler = MinMaxScaler()
# #scaler=StandardScaler()
# #scaler=RobustScaler()
# #scaler=MaxAbsScaler()
# scaler.fit(x1)
# scaler.fit(x2)
# x11=scaler.transform(x1)
# x22=scaler.transform(x2)


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


x1_co, y1_co = split_xy3(x11, 5, 2)
x2_ko, y2_ko = split_xy3(x22, 5, 2)



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_co, x2_ko, y1_co, y2_ko ,train_size=0.9, random_state=66)

print(x1_train.shape, x1_test.shape)    # (382, 5, 3) (96, 5, 3)
print(x2_train.shape, x2_test.shape)    # (382, 5, 3) (96, 5, 3)
print(y1_train.shape, y1_test.shape)    # (382, 2) (96, 2)
print(y2_train.shape, y2_test.shape)    # (382, 2) (96, 2)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델
input1 = Input(shape=(5, 3))
dense1 = LSTM(128, activation='relu')(input1)
dense2 = Dense(64, activation='linear')(dense1)
dense3 = Dropout(0.4)(dense2)
dense4 = Dense(32, activation='linear')(dense3)
dense5 = Dense(32, activation='linear')(dense4)
dense6 = Dense(32, activation='linear')(dense5)
dense7 = Dense(16, activation='linear')(dense6)
dense8 = Dense(8, activation='linear')(dense7)
output1 = Dense(16, activation='linear')(dense8)


#2-2. 모델
input2 = Input(shape=(5, 3))
dense11 = LSTM(128, activation='relu')(input2)
dense21 = Dense(64, activation='linear')(dense11)
dense31 =Dropout(0.4)(dense21)
dense41 = Dense(32, activation='linear')(dense31)
dense51 = Dense(32, activation='linear')(dense41)
dense61 = Dense(32, activation='linear')(dense51)
dense71 = Dense(16, activation='linear')(dense61)
dense81 = Dense(8, activation='linear')(dense71)
output2 = Dense(16, activation='linear')(dense81)



from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate()([output1, output2])#, axis=1)  # axis=0 y축방향 병합 (200,3)


#2-3 output모델1
output21 = Dense(32)(merge1)
output22 = Dense(32)(output21)
output23 = Dense(16)(output22)
output24 = Dense(8, activation='relu')(output23)
last_output1 = Dense(1)(output24)

#2-3 output모델2
output31 = Dense(32)(merge1)
output32 = Dense(32)(output31)
output33 = Dense(16)(output32)
output34 = Dense(8, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model = Model(inputs=[input1, input2], outputs= [last_output1, last_output2])


model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 20 , mode = 'auto', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1220_lastcost7.hdf6')

model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=1000, batch_size=10, validation_split=0.1, verbose=1, callbacks=[es])#,mcp]) 

# model.save_weights("./_save/co_ko_1_save_weights.h5")

#model = load_model("./_ModelCheckPoint/co_ko_1220_lastcost5.hdf5")

#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test])
print('loss :', loss) #loss :
a = X11[-5:]
b = X22[-5:]
a = a.reshape(1,5,3)
b = b.reshape(1,5,3)
y1_predict, y2_predict = model.predict([a, b])
print('코로나환자예상수 : ', y1_predict[0][-1])
print('코스피예상지수 : ', y2_predict[0][-1])

y1_pred, y2_pred = model.predict([x1_co, x2_ko])


# print(y1_pred[:5])

# '''
# 코로나환자수 [3019]
# 코스피지수  [2988.77]

# 코스피예상지수 :  [3068.3394]

# 코스피예상지수 :  [2978.539]

# 코스피예상지수 :  [2834.2913]

# 코스피예상지수 :  [3095.1743]

# 코스피예상지수 :  [2987.2097]

# 코스피예상지수 :  [2930.443]

# 코스피예상지수 :  3040.0396

# 코스피예상지수 :  2927.8196

# 코스피예상지수 :  3064.6743

# 코스피예상지수 :  [3059.4219

# 10번 평균치 2998
# '''
aa=x2[4:-2]
train_data=aa['현재지수']
valid_data=aa
print(train_data.shape)
print(valid_data.shape)
valid_data['Predictions']=y2_pred
plt.plot(train_data)
plt.plot(valid_data["Predictions"])
plt.show()