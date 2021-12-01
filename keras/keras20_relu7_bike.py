import numpy as np #숫자들의모임.
import pandas as pd  #다들어감
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


#1. 데이터
path = "./_data/bike/"
train_raw= pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path +'sampleSubmission.csv')

x = train_raw.drop(['datetime','casual','registered','count'], axis=1)
test_file = test_file.drop(['datetime'], axis=1)

print(x.columns)

train = train_raw.copy()
train['datetime'] = pd.to_datetime(train['datetime'])
train.dtypes
count_q1 = np.percentile(train['count'], 25)
count_q1
# 'count' 데이터에서 전체의 75%에 해당하는 데이터 조회
count_q3 = np.percentile(train['count'], 75)
count_q3
# IQR = Q3 - Q1
count_IQR = count_q3 - count_q1
count_IQR
train_clean = train[(train['count'] >= (count_q1 - (1.5 * count_IQR))) & (train['count'] <= (count_q3 + (1.5 * count_IQR)))]

# datetime -> integer 타입으로 변환하는 사용자 정의 함수
def to_integer(datetime):
  return 10000 * datetime.year + 100 * datetime.month + datetime.day


y = train_raw['count']


#데이타 로그변환 : y 값이 한쪽으로 몰릴시 통상 사용
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)


#2. 모델구성

model = Sequential()
model.add(Dense(250, input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(14, activation='linear'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #rms
model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.2, verbose=1) # batch_size=default 는 32이다.

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_pred = model.predict(x_test)

rmse = RMSE(y_test, y_pred)
print('RMSE :', rmse) #loss :


r2 = r2_score(y_test, y_pred)
print('r2score :', r2) 


###################제출용 제작############################
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + 'submit_test10.csv', index = False)


'''
loss : 1.4842497110366821
RMSE : 1.2182978840821634
r2score : 0.24255754427571363

==============================

# MaxAbsScaler
loss : 1.4563162326812744
RMSE : 1.2067792783898472
r2score : 0.25681257434900484

layer에 relu 반영시 값이 큰 차이가 없다.
loss : 1.4526137113571167
RMSE : 1.2052443360375928
r2score : 0.2587019411902052

'''

'''
# MinMaxScaler                       #layer 에 relu 반영시

loss : 1.4626810550689697           loss : 1.3933570384979248
RMSE : 1.2094133980701014           RMSE : 1.180405454428457
r2score : 0.2535646213860472        r2score : 0.28894191371109745

# StandardScaler
loss : 1.480304479598999            loss : 1.437126874923706
RMSE : 1.2166776700715956           RMSE : 1.198802188629926
r2score : 0.2445708496519564        r2score : 0.2666053817354287

# RobustScaler
loss : 1.4949843883514404            loss : 1.3558378219604492
RMSE : 1.2226955556063406            RMSE : 1.1644046876778524
r2score : 0.23707941768263563        r2score : 0.3080884907137209
# MaxAbsScaler
loss : 1.4563162326812744            loss : 1.4526137113571167
RMSE : 1.2067792783898472            RMSE : 1.2052443360375928
r2score : 0.25681257434900484        r2score : 0.2587019411902052


'''