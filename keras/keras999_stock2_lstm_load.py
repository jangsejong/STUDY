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

#1 데이터
path = "D:\\Study\\_data\\bit\\stock\\"
samsung = pd.read_csv(path +"삼성전자.csv", thousands=",", encoding='cp949')
kiwoom = pd.read_csv(path +"키움증권.csv", thousands=",", encoding='cp949')

# 삼성주식의 액면 분할 전시점을 날려주며 행을 맞춰준다.
samsung1 = samsung.drop(range(30,1120), axis=0)
kiwoom1 = kiwoom.drop(range(30,1060), axis=0)

#과거순으로 행을 역순 시켜 준다.
samsung2 = samsung1.loc[::-1].reset_index(drop=True)
kiwoom2 = kiwoom1.loc[::-1].reset_index(drop=True)


x1 = samsung2.drop(columns=['일자','Unnamed: 6','등락률', '고가', '시가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '거래량', '기관', '외국계'], axis=1) 
x2 = kiwoom2.drop(columns =['일자','Unnamed: 6','등락률', '고가', '시가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '거래량', '기관', '외국계'], axis=1) 
x1 = np.array(x1)
x2 = np.array(x2)
print(x1.shape, x2.shape) #(893, 4) (893, 4)


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

'''
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
'''
model = load_model("./_ModelCheckPoint/ss_ki_1220_lastcost5.hdf5")
#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1_test, x2_test])
print('삼성예측값 : ', y1_pred[-1])
print('키움예측값 : ', y2_pred[-1])
print(y1_pred[:5])

'''
삼성 78,000원
키움 109,500원

ss_ki_1220_lastcost5
삼성예측값 :  [76598.695]
키움예측값 :  [107669.79]

'''