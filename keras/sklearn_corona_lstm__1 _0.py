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

covid_19 = pd.read_csv(path +"코로나바이러스감염증-19_확진환자_발생현황_220110.csv", thousands=",", encoding='cp949')
#kospi = pd.read_csv(path +"코스피지수(202001~202112).csv", thousands=",", encoding='cp949')

# print(covid_19.head())
# print(kospi.head())

#데이터 전처리

#covid_19.csv 에서 결측치 칼럼이 확인되어 삭제해주었다.

# covid_19 = covid_19.drop(covid_19.index[range(0,1)] ,axis=0)
covid_19 = covid_19.drop(covid_19.columns[range(5,8)], axis=1)
# covid_19 = covid_19.replace("-","0")



#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.
# kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)




# 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
# kospi['new_Date'] = pd.to_datetime(kospi['일자'])

covid_19.drop('일자', axis = 1, inplace=True)
covid_19.set_index('new_Date', inplace=True)

# kospi.drop('일자', axis = 1, inplace=True)
# kospi.set_index('new_Date', inplace=True)
# print(x1.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
x1 = covid_19.drop(columns=[], axis=1) 
# x2 = kospi.drop(columns =['대비','등락률(%)', '주가이익비율', '주가자산비율', '거래량(천주)', '상장시가총액(백만원)', '거래대금(백만원)','배당수익률(%)'], axis=1) 

# for col1 in x1.columns:
#     n_nan1 = x1[col1].isnull().sum()
#     if n_nan1>0:
#       msg1 = '{:^20}에서 결측치 개수 : {}개'.format(col1,n_nan1)
#       print(msg1)
#     else:
        #   print('결측치가 없습니다.')    
# for col2 in x2.columns:
#     n_nan2 = x2[col2].isnull().sum()
#     if n_nan2>0:
#       msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
#       print(msg2)
#     else:
#         #   print('결측치가 없습니다.')  

# print(x1.head())
# print(x2.head())


#주식장이 열리지 않는 주말에 해당하는 행을 삭제하여 준다.
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SAT'),axis=0)
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SUN'),axis=0)

import holidays
import pandas as pd


kr_holidays = holidays.KR(years=[2020,2021])#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 삭제하여 준다.
import holidays
'''
{datetime.date(2020, 1, 1): "New Year's Day", datetime.date(2020, 1, 24): "The day preceding of Lunar New Year's Day", datetime.date(2020, 1, 25): "Lunar New Year's Day", datetime.date(2020, 1, 26): 
"The second day of Lunar New Year's Day", datetime.date(2020, 1, 27): "Alternative holiday of Lunar New Year's Day", datetime.date(2020, 3, 1): 'Independence Movement Day', 
datetime.date(2020, 4, 30): 'Birthday of the Buddha', datetime.date(2020, 5, 5): "Children's Day", datetime.date(2020, 5, 1): 'Labour Day', datetime.date(2020, 6, 6): 'Memorial Day', 
datetime.date(2020, 8, 15): 'Liberation Day', datetime.date(2020, 9, 30): 'The day preceding of Chuseok', datetime.date(2020, 10, 1): 'Chuseok', datetime.date(2020, 10, 2): 'The second day of Chuseok', 
datetime.date(2020, 10, 3): 'National Foundation Day', datetime.date(2020, 10, 9): 'Hangeul Day', datetime.date(2020, 12, 25): 'Christmas Day', datetime.date(2020, 8, 17): 'Alternative public holiday'}
{datetime.date(2021, 1, 1): "New Year's Day", datetime.date(2021, 2, 11): "The day preceding of Lunar New Year's Day", datetime.date(2021, 2, 12): "Lunar New Year's Day", datetime.date(2021, 2, 13): 
"The second day of Lunar New Year's Day", datetime.date(2021, 3, 1): 'Independence Movement Day', datetime.date(2021, 5, 19): 'Birthday of the Buddha', datetime.date(2021, 5, 5): "Children's Day", 
datetime.date(2021, 5, 1): 'Labour Day', datetime.date(2021, 6, 6): 'Memorial Day', datetime.date(2021, 8, 15): 'Liberation Day', datetime.date(2021, 8, 16): 'Alternative holiday of Liberation Day', 
datetime.date(2021, 9, 20): 'The day preceding of Chuseok', datetime.date(2021, 9, 21): 'Chuseok', datetime.date(2021, 9, 22): 'The second day of Chuseok', datetime.date(2021, 10, 3): 'National Foundation Day', 
datetime.date(2021, 10, 4): 'Alternative holiday of National Foundation Day', datetime.date(2021, 10, 9): 'Hangeul Day', datetime.date(2021, 10, 11): 'Alternative holiday of Hangeul Day', datetime.date(2021, 12, 25): 'Christmas Day'}
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


print(x1.head())
# print(x2.head())

# print(x1.info)
# print(x2.info)
# print(x1.head)
# print(x2.head)
# print(x1.shape)
# print(x2.shape)


x11 = np.array(x1)
# x22 = np.array(x2)


import matplotlib
import pandas as pd
import numpy as np
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import matplotlib_inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# # 설치된 폰트 출력
# font_list = [font.name for font in fm.fontManager.ttflist]
# print(font_list)

#
import matplotlib
from matplotlib import font_manager, rc
import platform


#matplotlib 에서 사용하는 폰트를 한글 지원이 가능한 것으로 바꾸는 코드
if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:    
# Mac 인 경우
    rc('font', family='AppleGothic')
    
matplotlib.rcParams['axes.unicode_minus'] = False   
#그래프에서 마이너스 기호가 표시되도록 하는 설정입니다. 

plt.figure(figsize=(10,8))
plt.title("CORONA of Features", y = 1.05, size = 15)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# plt.xticks(rotation = + 20 )
# sns.barplot(x="일자", y="현재지수",data= kospi)

sns.heatmap(data= covid_19.corr(), square=True, annot=True, cbar=True)
plt.show()    
# plt.figure(figsize=(16, 9))
# sns.lineplot(y=x2['현재지수'], x=x2.index)
# plt.xlabel('new_Date')
# plt.ylabel('현재지수')
# plt.show()

# # 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
# covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
# # kospi['new_Date'] = pd.to_datetime(kospi['일자'])

# covid_19.drop('일자', axis = 1, inplace=True)
# covid_19.set_index('new_Date', inplace=True)

# kospi.drop('일자', axis = 1, inplace=True)
# kospi.set_index('new_Date', inplace=True)
# print(kospi.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
x1 = covid_19.drop(columns=[], axis=1) 
# x2 = kospi.drop(columns =['대비','등락률(%)', '상장시가총액(백만원)', '주가이익비율', '거래량(천주)', '배당수익률(%)', '거래대금(백만원)'], axis=1) 


#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 삭제하여 준다.
import holidays
kr_holidays = holidays.KR(years=[2020,2021])
# print(kr_holidays)

# x22 = np.array(x2)

from sklearn.preprocessing import MinMaxScaler

x1.sort_index(ascending=False).reset_index(drop=True)

scaler = MinMaxScaler()
scale_cols = ['계(명)', '국내발생(명)', '해외유입(명)', '누적환자수']
df_scaled = scaler.fit_transform(x1[scale_cols])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

df_scaled
TEST_SIZE = 200
WINDOW_SIZE = 20

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

from sklearn.model_selection import train_test_split

feature_cols = [ '국내발생(명)', '해외유입(명)', '누적환자수']
label_cols = ['계(명)']

train_feature = train[feature_cols]
train_label = train[label_cols]

train_feature, train_label = make_dataset(train_feature, train_label, 20)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
x_train.shape, x_valid.shape

test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature.shape, test_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape


# print(x2.info())

# def split_xy3(dataset, time_steps, y_column):                 
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps                    # 0 + 5 = 5
#         y_end_number = x_end_number + y_column      # 5 + 2 = 7
        
#         if y_end_number > len(dataset):
#             break
             
#         tmp_x = dataset[i:x_end_number, 1:]                #0:5,  1:
#         tmp_y = dataset[x_end_number:y_end_number, 0]    #5:7, 0
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)


# x2_ko, y2_ko = split_xy3(x22, 5, 2)

# from sklearn.model_selection import train_test_split

# x2_train, x2_test, y2_train, y2_test = train_test_split(x2_ko, y2_ko ,train_size=0.8, random_state=66)

# print(x1_train.shape) #(20, 5, 1)




#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(LSTM(64, activation='linear', input_shape=(20, 3)))
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mean_squared_error', optimizer='adam')#, metrics=['mae']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 20 , mode = 'auto', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1220_lastcost7.hdf6')

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])#,mcp]) 

# model.save_weights("./_save/keras999_1_save_weights.h5")

#model = load_model("./_ModelCheckPoint/ss_ki_1220_lastcost5.hdf5")

#4. 평가, 예측
# loss = model.evaluate ([x2_test], [y2_test], batch_size=1)
# print('loss :', loss) #loss :
# y2_pred = model.predict([x2_test])
# print('코스피예상지수 : ', y2_pred[-1])
# print(y2_pred[:5])

pred = model.predict(test_feature)

pred.shape

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

plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()

# train_data=x2[:388] 
# valid_data=x2[388:] 
# valid_data['Predictions']=y2_pred 
# plt.plot(train_data["현재지수"]) 
# plt.plot(valid_data[['현재지수',"Predictions"]])
# plt.show()
