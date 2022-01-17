#1. 데이터
import numpy as np
x1 = np.array([range(100), range(301,401)])   #삼성 저가,종가
x2 = np.array([range(101,201), range(411,511), range(100,200)]) #미국선물 시가,고가,종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array(range(1001,1101)) # 삼성전자 종가
y2 = np.array(range(101,201))
#y = np.transpose(y)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
# print(x1.shape, x2.shape, y1.shape, y2.shape)  #(100, 2) (100, 3) (100,) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2,train_size=0.8, random_state=66)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

#2-2. 모델
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2],axis=1)  # axis=0 y축방향 병합 (200,3)
# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output = Dense(1)(merge3)
# model = Model(inputs=[input1, input2], outputs= last_output)

# model.summary()

# merge1 = Concatenate()([output1, output2])#,axis=1)
# merge2 = Dense(10, activation='relu')(merge1)
# merge3 = Dense(7)(merge2)
# last_output = Dense(1)(merge3)
# model = Model(inputs=[input1, input2], outputs= last_output)

#2-3 output모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output모델2
output31 = Dense(2)(merge1)
output32 = Dense(10)(output31)
output33 = Dense(25, activation='relu')(output32)
last_output2 = Dense(1)(output33)

model = Model(inputs=[input1, input2], outputs= [last_output1, last_output2])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) #rms
model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose=1) 

#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :
y1_pred, y2_pred = model.predict([x1_test, x2_test])

'''
mae
#loss : [0.05003878474235535, 0.031221460551023483, 0.018817326053977013, 0.08458862453699112, 0.06721572577953339]
'''