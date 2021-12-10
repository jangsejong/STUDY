from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
import time

dataset  = load_breast_cancer()

x = dataset.data
y = dataset.target

# print(x.shape, y.shape) #(569,30) (569,)
# print(np.unique(y)) #----> [0, 1] : 배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = StandardScaler()

n = x_train.shape[0]# 
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #455,30

x_train = x_train_transe.reshape(n,3,5,2) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,3,5,2)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, 
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding ='same', activation='relu'))#<------------
# model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),padding ='same', activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
# print("==========================================★")
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 1000
patience_num = 20
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print("epochs :",epoch)

    