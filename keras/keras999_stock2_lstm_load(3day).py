from google.protobuf.descriptor_pool import DescriptorPool
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
submit_file = pd.read_csv(path +"제출용.csv", thousands=",", encoding='cp949')
# 삼성주식의 액면 분할 전시점을 날려주며 행을 맞춰준다.
samsung1 = samsung.drop(range(11,1120), axis=0)
kiwoom1 = kiwoom.drop(range(11,1060), axis=0)

#과거순으로 행을 역순 시켜 준다.
samsung2 = samsung1.loc[::-1].reset_index(drop=True)
kiwoom2 = kiwoom1.loc[::-1].reset_index(drop=True)
'''
print(samsung.head())

      일자      시가      고가      저가      종가 전일비    Unnamed: 6   등락률          거래량      금액(백만)  신용비       개인       기관     외인(수량)    외국계     프로그램 외인비
4  2021/12/13  77,200   78,300    76,500   76,800   ▼       -100        -0.13        15,038,750  1,163,285    0.13    -181,359     184,966   -151,301  -1,388,477   -606,534  51.75
'''
x1 = samsung2.drop(columns=['일자','Unnamed: 6','등락률', '거래량', '종가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '외국계', '기관', '저가'], axis=1) 
x2 = kiwoom2.drop(columns =['일자','Unnamed: 6','등락률', '거래량', '종가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '외국계', '기관', '저가'], axis=1) 
x1 = np.array(x1)
x2 = np.array(x2)
print(x1.shape, x2.shape) #(30, 2) (30, 2)
print(samsung.info())

def split_xy3(dataset, time_steps, y_column):                 
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps                 # 0 + 5 = 5
        y_end_number = x_end_number + y_column - 0    # 5 + 3 -0 = 8
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]                # 0 : 5
        tmp_y = dataset[x_end_number-1:y_end_number, 0]    # 5 : 8 , 0
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x1_ss, y1_ss = split_xy3(x1, 5, 3)
x2_ki, y2_ki = split_xy3(x2, 5, 3)

# print(x1_ss[-1])



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_ss, x2_ki, y1_ss, y2_ki ,train_size=0.8, random_state=66)

print(x1_train.shape) #(58, 5, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델
input1 = Input(shape=(5, 1))
dense1 = LSTM(10, activation='linear', name='dense1')(input1)
dense2 = Dense(40, activation='linear', name='dense2')(dense1)
dense3 = Dense(20, activation='linear', name='dense3')(dense2)
output1 = Dense(10, activation='linear', name='output1')(dense3)


#2-2. 모델
input2 = Input(shape=(5, 1))
dense11 = LSTM(10, activation='linear', name='dense11')(input2)
dense21 = Dense(40, activation='linear', name='dense21')(dense11)
dense31 = Dense(20., activation='linear', name='dense31')(dense21)
output2 = Dense(10, activation='linear', name='output2')(dense31)


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
model.compile(loss='mse', optimizer='adam', metrics=['mse']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 50 , mode = 'auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1222_Trafevol3.hdf5')

model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es,mcp]) 

#model.save_weights("./_save/keras999_1_save_weights.h5")
#model = load_model('./_ModelCheckPoint/ss_ki_1222_Trafevol1.hdf5')

#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1_test, x2_test])

#print(y1_pred[:5])
pred1 = np.array(x1[-5:,1:]).reshape(1,5,1)
pred2 = np.array(x2[-5:,1:]).reshape(1,5,1)
print(x1[-5:,1:])
result1, result2 = model.predict([pred1, pred2])


print('삼성예측값 : ', y1_pred[-1][-1])
print('키움예측값 : ', y2_pred[-1][-1])

#submit_file.to_csv(path+'2day(거래량).csv', index=False)
'''
삼성 78000    72000~83000
키움 108500   100000~117000

ss_ki_1222_Trafevol1
삼성예측값 :  77288.805
키움예측값 :  107841.56

ss_ki_1222_Trafevol2
삼성예측값 :  77055.95
키움예측값 :  119642.91

ss_ki_1222_Trafevol3
삼성예측값 :  80549.375
키움예측값 :  110761.734

ss_ki_1222_Trafevol4
삼성예측값 :  77097.62
키움예측값 :  107414.85

'''