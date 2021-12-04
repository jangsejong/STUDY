import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import accuracy
from pandas import get_dummies


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

x = train.drop(['id', 'quality'], axis =1)
y = train['quality']

#print(type(train)) #<class 'pandas.core.frame.DataFrame'>

 
#print(train.head(3))
le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
# print(x)
# #print(x.type.value_counts())
from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
#y = to_categorical(y)

test_file = test_flie.drop(['id'], axis=1)
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

#데이타 로그변환 : y 값이 한쪽으로 몰릴시 통상 사용
#y = np.log1p(y)c

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

y = train['quality']
# print(y.unique())                # [6 7 5 8 4]
y = get_dummies(y)

#2. 모델구성
input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(1)(dense2)
dense4 = Dense(2)(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(6)(dense5)
ouput1 = Dense(5, activation='sigmoid')(dense6)
model = Model(inputs=input1, outputs=ouput1)


model.save("./_save/Dacon_wine2.h5")

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=9, validation_split=0.2, callbacks=[es])


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
#print("epochs :",epochs)


test_flie['type'] = le.transform(test_flie['type'])


scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
result_recover = np.argmax(result, axis =1).reshape(-1,1)
sample_submission['quality'] = result_recover
#print(result_recover)
#print(np.unique(result_recover))
# # print(submission[:10])
sample_submission.to_csv(path+"test_01.csv", index = False)
#print(result_recover)
'''
loss :  1.0884289741516113
accuracy :  0.5409582853317261

'''