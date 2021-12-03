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

test_flie = pd.read_csv(path + "test.csv") #index_col=0,  header=0)  
sample_submission = pd.read_csv(path + "sample_submission.csv", index_col=0,  header=0)

#print(test_file.shape) #(3231, 13)
submit_file = pd.read_csv(path +'sample_submission.csv')
#print(sample_submission.shape) #(3231, 1)
#print(train.columns, test_file.columns)   #['quality']

x = train.drop(['quality'], axis =1)
y = train['quality']

#print(type(train)) #<class 'pandas.core.frame.DataFrame'>

 
#print(train.head(3))
le = LabelEncoder()
le.fit(train.type)
x['type'] = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
# print(x)

from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y)


#데이타 로그변환 : y 값이 한쪽으로 몰릴시 통상 사용
#y = np.log1p(y)c

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2. 모델구성

input1 = Input(shape=(x.shape[1],))
dense1 = Dense(60)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(18)(dense2)
dense4 = Dense(6, activation='relu')(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(2)(dense5)
ouput1 = Dense(y.shape[1], activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
#print("epochs :",epochs)


test_flie['type'] = le.transform(test_flie['type'])


scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
result_recover = np.argmax(y, axis =1).reshape(-1,1)
sample_submission['quality'] = result_recover

# # print(submission[:10])
sample_submission.to_csv(path+"sampleHR_MinMaxScaler.csv", index = False)
print(result_recover)
