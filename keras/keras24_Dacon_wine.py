import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.metrics import accuracy


def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


#1. 데이터
path = "../../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")# index_col=0,  header=0) # 인덱스 조절하여 1열 삭제, 헤드 조절하여 행 선정
#print(train)
#print(train.shape) #(3231, 14)

test_file = pd.read_csv(path + "test.csv") #index_col=0,  header=0)  
sample_submission = pd.read_csv(path + "sample_submission.csv", index_col=0,  header=0)

#print(test_file.shape) #(3231, 13)
submit_file = pd.read_csv(path +'sample_submission.csv')
#print(sample_submission.shape) #(3231, 1)
print(train.columns, test_file.columns)   #['quality']

x = train.drop(['quality'], axis =1)
y = train['quality']

#print(type(train)) #<class 'pandas.core.frame.DataFrame'>

 
#print(train.head(3))
le = LabelEncoder()
le.fit(train.type)
x_type = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
x['type'] = x_type
# print(x)

from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y)

#데이타 로그변환 : y 값이 한쪽으로 몰릴시 통상 사용
#y = np.log1p(y)c

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

input1 = Input(shape=(x.shape[1],))
dense1 = Dense(30)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(18)(dense2)
dense4 = Dense(8, activation='relu')(dense3)
dense5 = Dense(3)(dense4)
dense6 = Dense(2)(dense5)
ouput1 = Dense(y.shape[1], activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=15, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=10000, batch_size=13, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_pred = model.predict(x_test)

rmse = RMSE(y_test, y_pred)
print('RMSE :', rmse) #loss :


r2 = r2_score(y_test, y_pred)
print('r2score :', r2) 


'''


###################제출용 제작############################
#1
loss : [0.06394559890031815, 0.5425038933753967]
RMSE : 0.25287464
r2score : 0.055973064387809864

#2
loss : [0.06334538757801056, 0.5625966191291809]
RMSE : 0.2516851
r2score : 0.050159268747012674

#3
loss : [1.0120439529418945, 0.5486862659454346]
RMSE : 0.2517688
r2score : 0.05500168355967101

#4
loss : [0.06246519088745117, 0.5548686385154724]
RMSE : 0.2499304
r2score : 0.06689176944257778

#5
loss : [0.06355590373277664, 0.5765069723129272]
RMSE : 0.25210294
r2score : 0.0598737408185788

$6
loss : [0.062167856842279434, 0.5440494418144226]
RMSE : 0.24933483
r2score : 0.06662338098186814

#7
loss : [0.06592332571744919, 0.5255023241043091]
RMSE : 0.2567554
r2score : 0.0381328070693481

$8
loss : [0.06366151571273804, 0.5595054030418396]
RMSE : 0.25231236
r2score : 0.041760525511112596

$9


'''



submit_file.to_csv(path + 'submit_test9.csv', index = False)
