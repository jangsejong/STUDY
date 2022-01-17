import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))  

path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path +"train.csv")
# print(train) 10886,12

test_flie = pd.read_csv(path + "test.csv") #### 제출용 test는 시험용!
# print(test) 6493,9
submission = pd.read_csv(path+"sampleSubmission.csv") #제출할 값
# print(submission) 6493,2

y = train['count']
x = train.drop(['casual','registered','count'], axis =1) #

x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)


test_flie['datetime'] = pd.to_datetime(test_flie['datetime'])
test_flie['year'] = test_flie['datetime'].dt.year
test_flie['month'] = test_flie['datetime'].dt.month
test_flie['day'] = test_flie['datetime'].dt.day
test_flie['hour'] = test_flie['datetime'].dt.hour
test_flie = test_flie.drop('datetime', axis=1)


# 로그변환
y = np.log1p(y) #----------> 로그변환 하면서 +1 해줌

x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2 모델구성#        
deep_len = [100, 80, 60, 50, 40, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]

input1 = Input(shape=(x.shape[1],))
dense1 = Dense(deep_len[0])(input1)
dense2 = Dense(deep_len[1])(dense1)
dense3 = Dense(deep_len[2])(dense2)
dense4 = Dense(deep_len[3])(dense3)
dense5 = Dense(deep_len[4])(dense4)
dense6 = Dense(deep_len[5])(dense5)
dense7 = Dense(deep_len[6])(dense6)
dense8 = Dense(deep_len[7])(dense7)
dense9 = Dense(deep_len[8])(dense8)
dense10 = Dense(deep_len[9])(dense9)
dense11 = Dense(deep_len[10])(dense10)
dense12 = Dense(deep_len[11])(dense11)
dense13 = Dense(deep_len[12])(dense12)
dense14 = Dense(deep_len[13])(dense13)
dense15 = Dense(deep_len[14])(dense14)
dense16 = Dense(deep_len[15])(dense15)
output1 = Dense(1)(dense16)

model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = "mse", optimizer = 'adam') 

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es])#batch_size =32 가 default
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss) 

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

rmse = RMSE(y_test, y_predict)
print("R2 : ", r2)
print("RMSE : ",rmse)
# print("epochs :",epoch)
print(deep_len)

acc= str(loss[1]).replace(".", "_")
model.save(f"./_save/kaggle_bike_{acc}.h5")

