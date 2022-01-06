import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
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

path = 'D:\\Study\\개인프로젝트\\데이터자료\\csv\\'

covid_19 = pd.read_csv(path +"코로나바이러스감염증_19_확진환자_현황.csv", thousands=",", encoding='cp949')
kospi = pd.read_csv(path +"코스피지수(202001~202112).csv", thousands=",", encoding='cp949')

# print(covid_19.head())
# print(kospi.head())

#covid_19.csv 에서 결측치 칼럼이 확인되어 삭제해주었으며, 필요없는 첫번째 누적 수치행을 삭제 해주었다.

covid_19 = covid_19.drop(covid_19.index[range(0,1)] ,axis=0)
covid_19 = covid_19.drop(covid_19.columns[range(5,10)], axis=1)
covid_19 = covid_19.replace("-","0")


#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.

kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)

# print(covid_19.head())
# print(kospi.info())


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import platform
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

x1 = covid_19.drop(columns=['일자'], axis=1) 
x2 = kospi.drop(columns =['일자','대비','등락률(%)', '배당수익률(%)', '주가이익비율', '주가자산비율', '거래량(천주)', '현재지수', '거래대금(백만원)'], axis=1) 
x1 = np.array(x1)
x2 = np.array(x2)
print(x1.shape, x2.shape) #(714, 4) (484, 4)

'''
def split_xy3(dataset, time_steps, y_column):                 
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x1_ss, y1_ss = split_xy3(x1, 5, 2)
x2_ki, y2_ki = split_xy3(x2, 5, 2)



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_ss, x2_ki, y1_ss, y2_ki ,train_size=0.8, random_state=66)

print(x1_train.shape) #(20, 5, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델
input1 = Input(shape=(5, 1))
dense1 = LSTM(10, activation='linear', name='dense1')(input1)
dense2 = Dense(4, activation='linear', name='dense2')(dense1)
dense3 = Dense(2, activation='linear', name='dense3')(dense2)
output1 = Dense(1, activation='linear', name='output1')(dense3)


#2-2. 모델
input2 = Input(shape=(5, 1))
dense11 = LSTM(10, activation='linear', name='dense11')(input2)
dense21 = Dense(4, activation='linear', name='dense21')(dense11)
dense31 = Dense(2, activation='linear', name='dense31')(dense21)
output2 = Dense(1, activation='linear', name='output2')(dense31)


from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate()([output1, output2])#, axis=1)  # axis=0 y축방향 병합 (200,3)


#2-3 output모델1
output21 = Dense(16)(merge1)
output22 = Dense(8)(output21)
output23 = Dense(4, activation='linear')(output22)
last_output1 = Dense(1)(output23)

#2-3 output모델2
output31 = Dense(16)(merge1)
output32 = Dense(8)(output31)
output33 = Dense(4, activation='linear')(output32)
last_output2 = Dense(1)(output33)

model = Model(inputs=[input1, input2], outputs= [last_output1, last_output2])



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 20 , mode = 'auto', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1220_lastcost7.hdf6')

model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])#,mcp]) 

model.save_weights("./_save/keras999_1_save_weights.h5")

#model = load_model("./_ModelCheckPoint/ss_ki_1220_lastcost5.hdf5")
#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1_test, x2_test])
print('삼성예측값 : ', y1_pred[-1])
print('키움예측값 : ', y2_pred[-1])
print(y1_pred[:5])

'''

