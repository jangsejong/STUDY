from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import time
#1.데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()


n = x_train.shape[0]# 
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #353,10

x_train = x_train_transe.reshape(n,5,2,1) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,5,2,1)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, input_shape = (5,2,1)))#
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
model.add(Dense(1))

#model.summary()#3,153
#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'mse', optimizer = opt) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 20
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 20)
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
