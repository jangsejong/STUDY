import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

#01. data

path = "./_data/bike/"
train_raw= pd.read_csv(path + 'train.csv')                    #(10886, 12)
test_file = pd.read_csv(path + "test.csv")                 #(6493, 9)    , features (casual,registered,count) 없음 
submit_file = pd.read_csv(path +'sampleSubmission.csv')    #(6493, 2)

#train[train.duplicated()] # 중복값 없음 
#test[test.duplicated()] # 중복값 없음
#train.isnull().sum() #결측값 없음
#test.isnull().sum() #결측값 없음

#train.info()
#print(train.info())  #object 모든 자료의 최상위형
#print(train.describe()) #std 표준편차  mean, 중위값과 평균값차이

train = train_raw.copy()
train['datetime'] = pd.to_datetime(train['datetime'])
train.dtypes

#train.info()  데이터의 타입을 확인하니 날짜 데이터가 object 타입이다. 날짜 데이터를 보다 쉽게 조회하기 위해 아래와 같이 datetime 타입으로 변경했다.
train.isnull().sum()
#print(train.isnull().sum())
#datetime 형태의 데이터는 선형 회귀에 사용할 수 없다(참고로 선형 회귀는 정수형 혹은 실수형 데이터를 다룬다

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second

#print(train['weather'].value_counts())   날씨의 특성별 대여량의 차이


# 요일 데이터 - 일요일은 0
train['dayofweek'] = train['datetime'].dt.dayofweek

#print(train.describe()) # describe() 함수를 통해 데이터의 기초 통계값을 쉽게 조회할 수 있다.

# 'count' 데이터에서 전체의 25%에 해당하는 데이터 조회
count_q1 = np.percentile(train['count'], 25)
count_q1

# 'count' 데이터에서 전체의 75%에 해당하는 데이터 조회
count_q3 = np.percentile(train['count'], 75)
count_q3

# IQR = Q3 - Q1
count_IQR = count_q3 - count_q1
count_IQR

# 이상치를 제외한(이상치가 아닌 구간에 있는) 데이터만 조회
train_clean = train[(train['count'] >= (count_q1 - (1.5 * count_IQR))) & (train['count'] <= (count_q3 + (1.5 * count_IQR)))]

# datetime -> integer 타입으로 변환하는 사용자 정의 함수
def to_integer(datetime):
  return 10000 * datetime.year + 100 * datetime.month + datetime.day


train_wo_outliers = train_clean[np.abs(train_clean["count"] - train_clean["count"].mean()) <= (3*train_clean["count"].std())]
  
# 데이터 타입 변경
#datetime_int = train_wo_outliers['datetime'].apply(lambda x: to_integer(x))
#train_wo_outliers['datetime'] = pd.Series(datetime_int)

train_sub = train_raw.drop(['casual','registered','count'], axis=1)
#test_file = test_file.drop(axis=1)

x = train_sub
y = train_raw['count']

# 데이터를 편리하게 분할해주는 라이브러리 활용
from sklearn.model_selection import train_test_split

# 훈련 데이터의 25%를 검증 데이터로 활용
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)




from sklearn.linear_model import LinearRegression


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(250, input_dim=8))
model.add(Dense(4, activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(1))

y = np.log1p(y)
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #rms

model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.2, verbose=1) # batch_size=default 는 32이다.

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y_test, y_pred)
print('RMSE :', rmse) #loss :


r2 = r2_score(y_test, y_pred)
print('r2score :', r2) 



###################제출용 제작############################
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + 'submit_test10.csv', index = False)