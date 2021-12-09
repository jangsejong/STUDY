from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model,  load_model


from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
#1 데이터
path = "D:\\_data\\dacon\\wine\\" 
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

y = train['quality']
x = train.drop(['id','quality'], axis =1) #

le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!
#--to_categorical은 빈부분을 채우니 주의 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#-------------------------
y = np.array(y).reshape(-1,1)
enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
enc.fit(y)
y = enc.transform(y).toarray()

# print(y.shape)


# print(np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#2 모델구성#        
node_num = [100, 80, 60, 40, 60, 40, 20, 30, 20, 10, 40, 30, 20, 10, 5, 2]
# node_num = [226, 172, 129, 76, 65, 52, 42, 33, 26, 17, 9, 8, 5, 3, 2]

input1 = Input(shape=(12,))
dense1 = Dense(30)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(5)(dense4)
dense6 = Dense(5)(dense5)
ouput1 = Dense(5, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=ouput1)
#3. 컴파일, 훈련
epoch = 10000
opt="Adamax"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 50
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k27_dacon_wine_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es,mcp], batch_size =12)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
# print("epochs :",epoch)


test_flie['type'] = le.transform(test_flie['type'])
test_flie = test_flie.drop(['id'], axis =1) #
test_flie = scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
# print(result[:5])

result_recover = np.argmax(result, axis = 1).reshape(-1,1) + 4
# print(result_recover[:5])
print(np.unique(result_recover)) # np.unique()

submission['quality'] = result_recover

# # print(submission[:10])

# print(result_recover)

acc_list = hist.history['accuracy']
acc = opt + "_acc_"+str(acc_list[-patience_num]).replace(".", "_")
print(acc)
# acc= str(loss[1]).replace(".", "_")
# model.save(f"./_save/keras24_dacon_save_model_{acc}.h5")
submission.to_csv(path+f"MCP_sampleHR_{acc}.csv", index = False)
'''
loss :  0.9937585592269897
accuracy :  0.5842349529266357

loss :  0.9977083206176758
accuracy :  0.5873261094093323

'''
