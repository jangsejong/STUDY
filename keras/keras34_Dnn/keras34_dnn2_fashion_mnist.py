from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import numpy as np
import time

#1 데이터

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

dim = x_train.shape[1]*x_train.shape[1]
out_node = len(np.unique(y_train))

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# model.add(Dense(128, input_shape = (28*28,)))
model.add(Dense(128, input_dim = dim))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(out_node, activation='softmax'))


#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 1000
patience_num = 50
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_fashion_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size =500)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("acc : ",loss[1])

'''
<CNN>>>
#acc : 0.8949999809265137
<DNN>
accuracy :  0.8636000156402588
'''