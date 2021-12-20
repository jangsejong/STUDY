import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import platform
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')




# def f1_score(answer, submission):
#     true = answer
#     pred = submission
#     score = metrics.f1_score(y_true=true, y_pred=pred)
#     return score



#1 데이터
path = "D:\\Study\\_data\\bit\\stock\\"
samsung = pd.read_csv(path +"삼성전자.csv", thousands=",", encoding='cp949')
kiwoom = pd.read_csv(path +"키움증권.csv", thousands=",", encoding='cp949')

#submission = pd.read_csv(path+"sample_submission.csv")



print(samsung.head())
'''
      일자      시가      고가      저가      종가 전일비    Unnamed: 6   등락률          거래량      금액(백만)  신용비       개인       기관     외인(수량)    외국계     프로그램 외인비
0  2021/12/17  76,800   77,700    76,800   77,500   ▼       -300        -0.39         6,871,456    531,146    0.00           0           0          0      17,465   -184,248  51.78
1  2021/12/16  78,500   78,500    77,400   77,800   ▲        200        0.26         11,996,128    934,244    0.13    -442,445    -261,746   -105,777     571,543    822,030  51.78
2  2021/12/15  76,400   77,600    76,300   77,600   ▲        600        0.78          9,584,939    738,592    0.14  -1,118,059    -654,764  1,095,947   1,946,258  1,706,254  51.79
3  2021/12/14  76,500   77,200    76,200   77,000   ▲        200        0.26         10,976,660    841,447    0.14     198,293  -1,487,295  1,005,909     804,186   -132,070  51.77
4  2021/12/13  77,200   78,300    76,500   76,800   ▼       -100        -0.13        15,038,750  1,163,285    0.13    -181,359     184,966   -151,301  -1,388,477   -606,534  51.75

# 주식 용어 정리
PER : Price Earning Ratio  기업은 매출과 당기순이익이 나오게 되는데 주가를 1주당 당기순이익으로 나눈것
PBR : Price on Book value Ratio 주가를 주당 순자산으로 나눈것
ROE : Return On Equity  기업이 투자한 자본을 활용해서 이익을 얼마나 냈는지 보는것
EPS : Earning Per Share 주식 1주당 이익이 얼마나 창출되었는지

주식지표와 주가는 별개임으로 해당 자료들은 사용 불가하다.
하지만 보조 지표로 참고한다면 리스크 대비가 가능하다.
'''

#sns.set_style('darkgrid')
#print(samsung.shape, kiwoom.shape)  #(1060, 17) (1160, 17)

# #결측치
# print(samsung.isnull().sum())
# print(kiwoom.isnull().sum())  
# 결측치는 존재하지 않았다.

# 변수의 타입 및 기초 통계량
# print(samsung.info())
# print(kiwoom.info())  
# 등락률,신용비,외인비를 제회한 모든 변수가 문자형 변수이다.

'''
 numerical variable
print(samsung.describe())
print(kiwoom.describe())
삼성        등락률          신용비          외인비
count    1060.000000  1060.000000  1060.000000
mean        0.060434     0.090330    54.646104
std         1.685135     0.073531     2.012295
min        -6.390000     0.000000    51.170000
25%        -0.962500     0.050000    52.607500
50%         0.000000     0.090000    54.915000
75%         1.000000     0.110000    56.620000
max        10.470000     1.780000    58.010000
키움        등락률          신용비          외인비
count    1160.000000  1160.000000  1160.000000
mean        0.055345     0.520086    23.283845
std         2.532978     0.343801     2.485882
min        -7.830000     0.000000    17.030000
25%        -1.450000     0.270000    21.800000
50%         0.000000     0.390000    23.520000
75%         1.350000     0.680000    25.122500
max        12.550000     1.490000    28.500000


# categorical variable
closing_price_sam = samsung['종가'].unique()
opening_price_sam = samsung['시가'].unique()
Trading_volume_sam = samsung['거래량'].unique()
#print(len(closing_price_sam))

closing_price_ki = kiwoom['종가'].unique()
opening_price_ki = kiwoom['시가'].unique()
Trading_volume_ki = kiwoom['거래량'].unique()
#print(len(closing_price_ki))

print(type(closing_price_sam))
print(type(closing_price_ki))

'''
# 삼성주식의 액면 분할 전시점을 날려주며 행을 맞춰준다.
samsung1 = samsung.drop(range(140,1120), axis=0)
kiwoom1 = kiwoom.drop(range(140,1060), axis=0)

#과거순으로 행을 역순 시켜 준다.
samsung2 = samsung1.loc[::-1].reset_index(drop=True)
kiwoom2 = kiwoom1.loc[::-1].reset_index(drop=True)

print(samsung.describe)

# print(samsung.shape, kiwoom.shape)  #(893, 17) (893, 17)

# samsung = samsung.apply(pd.to_numeric) # convert all columns of DataFrame
# kiwoom = kiwoom.apply(pd.to_numeric) # convert all columns of DataFrame

# print(samsung.info())
# print(kiwoom.info())  

x1 = samsung2.drop(columns=['일자','Unnamed: 6','등락률', '고가', '종가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '시가', '기관', '외국계', '저가'], axis=1) 
x2 = kiwoom2.drop(columns =['일자','Unnamed: 6','등락률', '고가', '종가', '금액(백만)', '전일비', '신용비', '개인', '외인(수량)', '프로그램', '외인비', '시가', '기관', '외국계', '저가'], axis=1) 
x1 = np.array(x1)
x2 = np.array(x2)
#print(x1.shape, x2.shape) #(893, 4) (893, 4)


# y1 = samsung['종가']
# y2 = kiwoom['종가']

#print(x1.shape, x2.shape, y1.shape, y2.shape)
def split_xy3(dataset, time_steps, y_column):                 
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   > 5
        y_end_number = x_end_number + y_column - 1    # 5 + 3 -1 > 7
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 0]                   # 0 : 5 , : -1   > 0행~4행, 마지막열뺀전부  > 
        tmp_y = dataset[x_end_number-1:y_end_number, 0]       # 5 - 1 : 7 , -1  > 마지막열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x1_ss, y1_ss = split_xy3(x1, 5, 3)
x2_ki, y2_ki = split_xy3(x2, 5, 3)

# print(x1_ss.shape, y1_ss.shape) (889, 4, 3) (889, 2)
# print(x2_ki.shape, y2_ki.shape) (889, 4, 3) (889, 2)




# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# #scaler = MaxAbsScaler()

# scaler.fit(x1_ss)
# scaler.fit(x2_ki)

# x1_ss_ = scaler.transform(x1_ss)
# x2_ki_ = scaler.transform(x2_ki)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_ss, x2_ki, y1_ss, y2_ki ,train_size=0.6, random_state=66)

print(x1_train.shape) #(19, 5, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델
input1 = Input(shape=(5, 1))
dense1 = LSTM(10, activation='relu', name='dense1')(input1)
dense2 = Dense(40, activation='linear', name='dense2')(dense1)
dense3 = Dense(20, activation='linear', name='dense3')(dense2)
output1 = Dense(10, activation='linear', name='output1')(dense3)
# model = Sequential()
# model.add(LSTM(8, input_length=10, input_dim=3))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# output1 = model.add(Dense(10, name='output1'))


#2-2. 모델
input2 = Input(shape=(5, 1))
dense11 = LSTM(10, activation='relu', name='dense11')(input2)
dense21 = Dense(40, activation='linear', name='dense21')(dense11)
dense31 = Dense(20, activation='linear', name='dense31')(dense21)
output2 = Dense(10, activation='linear', name='output2')(dense31)
# model = Sequential()
# model.add(LSTM(8, input_length=10, input_dim=3))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# output2 = model.add(Dense(10, name='output2'))

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate()([output1, output2])#, axis=1)  # axis=0 y축방향 병합 (200,3)
# model.summary()

# merge1 = Concatenate()([output1, output2])#,axis=1)
# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output = Dense(1)(merge3)
# model = Model(inputs=[input1, input2], outputs= last_output)

#2-3 output모델1
output21 = Dense(32)(merge1)
output22 = Dense(16)(output21)
output23 = Dense(8, activation='relu')(output22)
last_output1 = Dense(3)(output23)

#2-3 output모델2
output31 = Dense(32)(merge1)
output32 = Dense(16)(output31)
output33 = Dense(8, activation='relu')(output32)
last_output2 = Dense(3)(output33)

model = Model(inputs=[input1, input2], outputs= [last_output1, last_output2])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) #rms

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 50 , mode = 'auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=5, save_best_only=True, filepath='./_ModelCheckPoint/ss_ki_1220_lastcost10.hdf6')

model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es,mcp]) 



#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1, x2])
print('삼성예측값 : ', y1_pred[-1][-1])
print('키움예측값 : ', y2_pred[-1][-1])
print(y1_pred[:5])

'''
삼성 78,000원
키움 109,500원
ss_ki_1220_lastcost1
삼성예측값 :  [77770.85]       -230
키움예측값 :  [109066.91]      -433.09

ss_ki_1220_lastcost2        
삼성예측값 :  [77998.49]                    
키움예측값 :  [108415.336]          

ss_ki_1220_lastcost3
삼성예측값 :  [77921.875]
키움예측값 :  [108658.09]

ss_ki_1220_lastcost4
삼성예측값 :  [78018.96]
키움예측값 :  [109231.03]

ss_ki_1220_lastcost5
삼성예측값 :  [76598.695]
키움예측값 :  [107669.79]

ss_ki_1220_lastcost6
삼성예측값 :  [77998.336]
키움예측값 :  [106432.41]

ss_ki_1220_lastcost7(거래량)
삼성예측값 :  7745136.0
키움예측값 :  15842.258

5일치 11,367,410 / 48,705

ss_ki_1220_lastcost8(거래량)
삼성예측값 :  11139224.0
키움예측값 :  31802.123

ss_ki_1220_lastcost9(거래량)
삼성예측값 :  9605089.0
키움예측값 :  49075.27

ss_ki_1220_lastcost10(거래량)
삼성예측값 :  8966532.0
키움예측값 :  76041.04
'''