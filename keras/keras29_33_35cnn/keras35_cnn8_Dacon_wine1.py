from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
#1 데이터
path = "D:\\Study\\_data\\dacon\\wine\\"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

y = train['quality']
x = train.drop(['id','quality'], axis =1)
x = x.drop(['citric acid','pH','sulphates'], axis =1) #,<-------- 상관관계

le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])


y = np.array(y).reshape(-1,1)
enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
enc.fit(y)
y = enc.transform(y).toarray()
x = x.to_numpy()

###############
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

n = x_train.shape[0]
# x_train_reshape = x_train.reshape(n,-1) #
x_train_transe = scaler.fit_transform(x_train) #<------ 0.0 ~ 1.0
#################################################################
# print(x_train_transe.shape) # (2584,9)
x_train = x_train_transe.reshape(n,3,3,1) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,3,3,1)

#2 모델

model = Sequential()#6,6 ->2,2 -> 1 -> 4, 3
model.add(Conv2D(512, kernel_size=(2,2),padding ='valid',strides=1, activation='relu', 
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))#

#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(y.shape[1], activation = 'softmax')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련
opt = 'nadam'# 'Adaelta''adam',#
epoch = 10000
patience_num = 20


model.compile(loss = 'categorical_crossentropy', optimizer = opt , metrics=['accuracy'])
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 12)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
test_file['type'] = le.transform(test_file['type'])
test_file = test_file.drop(['id'], axis =1) #
test_file = test_file.drop(['citric acid','pH','sulphates'], axis =1) #,<-------- 상관관계
test_file = scaler.transform(test_file)

num = test_file.shape[0]
test_file = test_file.reshape(num,3,3,1) 
# test_file = test_file.to_numpy()

# ############### 제출용.
result = model.predict(test_file)
# print(result[:5])

result_recover = np.argmax(result, axis = 1).reshape(-1,1) + 4
# print(result_recover[:5])
print(np.unique(result_recover)) # np.unique()
submission['quality'] = result_recover

acc_list = hist.history['accuracy']
acc = opt + "_acc_"+str(acc_list[-patience_num]).replace(".", "_")
print(acc)
# acc= str(loss[1]).replace(".", "_")
model.save(f"./_save/_dacon_save_model_{acc}.h5")
submission.to_csv(f"./_save/{opt}_dacon_{acc}.csv", index = False)


'''
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''