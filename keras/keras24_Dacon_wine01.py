from tensorflow.keras.models import Sequential, Model,  load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies

#1 데이터
path = "../../_data/dacon/wine/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['id', 'quality'], axis =1)

le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)         # type column의 white, red > 0,1로 변환

from tensorflow.keras.utils import to_categorical

test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']      # [6 7 5 8 4]             
y = get_dummies(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.2, shuffle = True, random_state =66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2 모델구성
input1 = Input(shape=(12,))
dense1 = Dense(30)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(18)(dense2)
dense4 = Dense(6)(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(2)(dense5)
ouput1 = Dense(5, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode = 'auto', verbose=1, restore_best_weights=True)
#es = EarlyStopping(monitor='val_loss',patience=100, mode='min', verbose=1)

model.fit(x_train, y_train, epochs = 5000, batch_size = 30, validation_split=0.2, callbacks=[es])


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0])                      # List 형태로 제공된다
print("accuracy : ",loss[1])

################################ 제출용 ########################################
result = model.predict(test_file)
result_recover = np.argmax(result, axis=1).reshape(-1,1) + 4
submission['quality'] = result_recover

submission.to_csv(path + "010.csv", index = False)


'''
MinMax
loss :  1.01213538646698
accuracy :  0.5548686385154724
robust
loss :  1.01096773147583
accuracy :  0.5564142465591431
MaxAbs
loss :  1.007688045501709
accuracy :  0.5502318143844604
Standard
loss :  0.9866795539855957
accuracy :  0.5734157562255859


002
loss :  0.9862772822380066
accuracy :  0.5765069723129272

003
loss :  1.0153590440750122
accuracy :  0.5842349529266357

004
loss :  0.9992575645446777
accuracy :  0.5888717174530029

005
loss :  1.0092556476593018
accuracy :  0.5919629335403442

006
loss :  0.9349533915519714
accuracy :  0.6018518805503845

007
loss :  0.9498463273048401
accuracy :  0.6419752836227417

'''

