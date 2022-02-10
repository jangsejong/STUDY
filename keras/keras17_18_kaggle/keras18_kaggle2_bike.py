import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier

def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


#1. 데이터
path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv")# index_col=0,  header=0) # 인덱스 조절하여 1열 삭제, 헤드 조절하여 행 선정
#print(train)
#print(train.shape) #(10886, 12)

test_file = pd.read_csv(path + "test.csv") #index_col=0,  header=0)
#gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0,  header=0)

#print(test.shape) #(6493, 9)
submit_file = pd.read_csv(path +'sampleSubmission.csv')
#print(submit.shape) #(6493, 2)
#print(submit.columns)   #['datatime, count']


#print(type(train)) #<class 'pandas.core.frame.DataFrame'>
'''
#print(train.info())  #object 모든 자료의 최상위형
#print(train.describe()) #std 표준편차  mean, 
# 
#중위값과 평균값차이 link: https://blog.naver.com/conquer6022/221799258893

평균값(mean)

 N 개의 변량을 모두 더하여
그 개수로 나누어 놓은 숫자이다. 산술평균이라고도 한다.
N 개의 값을 크기 순으로 늘어놓았을 때 가장 가운데에 있는 숫자이다.
위 식에서는 (n+1)/2 = 16/2 = 8번째 있는 값인 5가 중위값이다.
이는 각 표본들의 격차가 워낙 클 때 주로 쓴다.
예를 들면, 직원이 100명인 회사에서 99명의 연봉 평균은 5천만 원인데 사장은 100억이었다고 해보자.
그럼 이 회사의 '평균'연봉은 1억 4851만 원이 된다.
따라서 이처럼 극단적인 값이 있는 경우 평균값보다 중위값이 더 유용하다.

==============================

결측값확인
#train.info()
.isna()
.isnull()
.notna()
.notnull()
#train[train.duplicated()] # 중복값 없음 
#test[test.duplicated()] # 중복값 없음

'''
#print(train.head(3))



x = train.drop(['datetime','casual','registered','count'], axis=1)
test_file = test_file.drop(['datetime'], axis=1)

print(x.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed'],
#      dtype='object')
#print(x.shape)  (10886, 8)
#submit_file = submit_file.drop(['datetime'], axis=1)
y = train['count']
#print(y.shape)  #(10886,)
'''
trainWind0=train.loc[train['windspeed']==0]
trainWindNot0=train.loc[train['windspeed']!=0]
def predict_windspeed(data): #(출력확인용)
    #풍속 예측에 사용되는 변수 
    wCol=['season','weather','humidity','temp','atemp']
    
    #풍속을 0인 것과 0이 아닌 것으로 구분
    dataWind0=data.loc[data['windspeed']==0]
    dataWindNot0=data.loc[data['windspeed']!=0]
    
    #랜덤포레스트 분류기 생성
    rfModel=RandomForestClassifier()
    dataWindNot0['windspeed']=dataWindNot0['windspeed'].astype("str")
    
    # wCol > 풍속학습 > 모델완성 
    rfModel.fit(dataWindNot0[wCol], dataWindNot0['windspeed']) #(학습대상, 학습자료)
    
    #학습한 모델로 풍속 0에 대한 데이터 예측 
    preValue=rfModel.predict(X=dataWind0[wCol])
    print(preValue)
    
    #풍속이 0인 것과 0이 아닌 것으로 분류 
    predictWind0=dataWind0
    predictWindNot0=dataWindNot0
    
    #예측값을 풍속이 0인 데이터에 대입
    predictWind0['windspeed']=preValue
    #풍속이 0이 아닌 데이터와 풍속이 0인 데이터 병합하여 data에 대입
    data=predictWindNot0.append(predictWind0)
    return data

def predict_windspeed2(data): #(실제데이터 보정용)
    #풍속 예측에 사용되는 변수 
    wCol=['season','weather','humidity','temp','atemp']
    
    #풍속을 0인 것과 0이 아닌 것으로 구분
    dataWind0=data.loc[data['windspeed']==0]
    dataWindNot0=data.loc[data['windspeed']!=0]
    
    #랜덤포레스트 분류기 생성
    rfModel=RandomForestClassifier()
    dataWindNot0['windspeed']=dataWindNot0['windspeed'].astype("str")
    
    # wCol > 풍속학습 > 모델완성 
    rfModel.fit(dataWindNot0[wCol], dataWindNot0['windspeed']) #(학습대상, 학습자료)
    
    #학습한 모델로 풍속 0에 대한 데이터 예측 
    preValue=rfModel.predict(X=dataWind0[wCol])
    print(preValue)
    
    #풍속이 0인 것과 0이 아닌 것으로 분류 
    predictWind0=dataWind0
    predictWindNot0=dataWindNot0
    
    #예측값을 풍속이 0인 데이터에 대입
    predictWind0['windspeed']=preValue
    #풍속이 0이 아닌 데이터와 풍속이 0인 데이터 병합하여 data에 대입
    data=predictWindNot0.append(predictWind0)

    # 행이름 reset(data만 따로 가져오기 위함)
    data.reset_index(inplace=True)
    data.drop('index',inplace=True, axis=1)
    return data
fig, ax1 = plt. subplots()
plt.sca(ax1)
plt.xticks(rotation=30)

#sns.countplot(data=train, x="windspeed", ax=ax1)
'''

#데이타 로그변환 : y 값이 한쪽으로 몰릴시 통상 사용
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)



#2. 모델구성

model = Sequential()
model.add(Dense(250, input_dim=8))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #rms
model.fit(x_train, y_train, epochs=12, batch_size=8, validation_split=0.2, verbose=1) # batch_size=default 는 32이다.

#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_pred = model.predict(x_test)

rmse = RMSE(y_test, y_pred)
print('RMSE :', rmse) #loss :


r2 = r2_score(y_test, y_pred)
print('r2score :', r2) 

'''
loss : 23708.40234375
RMSE : 153.9753502965621
r2score : 0.24991743293736246

--------
y 값 로그변환시
loss : 1.4570910930633545
RMSE : 1.207100372905771
r2score : 0.256417033659474


loss : 24206.748046875
RMSE : 155.58518026825467
r2score : 0.2424443511977793
'''


###################제출용 제작############################
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + 'submit_test10.csv', index = False)