from numpy import float32
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

import time


#1 데이터

(x_train, y_train),(x_test,y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# n = x_train.shape[0]
# x_train = x_train.reshape(n,-1)/255.
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.

# m = x_test.shape[0]
# x_test = x_test.reshape(m,-1)/255.
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.

print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28, 1) (10000,)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, LSTM

#2. 모델
# model = Sequential()
# # model.add(Dense(128, input_shape = (28*28,)))
# model.add(Dense(256, input_dim = 784))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(32))
# model.add(Dropout(0.4))
# model.add(Dense(16))
# model.add(Dropout(0.4))
# # model.add(Flatten(0.2))
# model.add(Dense(10, activation='softmax'))
model = Sequential()
##model.add(Conv2D(10, (2, 2), strides=1, padding='valid', input_shape=(10, 10, 1), activation='relu')) 
model.add(Conv2D(128 ,kernel_size=(10,8), padding='same', input_shape=(28, 28, 1)))     
model.add(Dropout(0.2))                     
# model.add(Conv2D(128,kernel_size=(8,6), padding='valid', activation='relu'))  #     (None, 13, 13, 5)                          
model.add(MaxPooling2D(pool_size=(8, 6), strides=1, padding='valid'))

model.add(Conv2D(64,kernel_size=(2,2), activation='relu'))  #     (None, 12, 12, 7) 
model.add(Dropout(0.2))                     
 
model.add(Conv2D(64,kernel_size=(2,2), activation='relu'))  #     (None, 11, 11, 7)  
model.add(Dropout(0.2))     
model.add(Conv2D(64,kernel_size=(2,2), activation='relu'))  #     (None, 11, 11, 7)  
model.add(Dropout(0.2))                     

                         
# model.add(Conv2D(10,kernel_size=(2,2), activation='relu')) #     (None, 10, 10, 10)                                                              
model.add(Flatten())                                       #     (None, 1000) 
# model.add(Reshape(target_shape=(100,10)))                  #     (None, 100, 10) 
# model.add(Conv1D(5, 2))                                    #     (None, 99, 5)  
# model.add(LSTM(15))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 30
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
# filepath = "./_ModelCheckPoint/"
# filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
# model_path = "".join([filepath,'k34_dnn_mnist_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =512, validation_split=0.1, callbacks=[es])#,mcp])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("acc : ",loss[1])

'''
learning_rate=0.01
시간 :  18.68 초
313/313 [==============================] - 1s 2ms/step - loss: 0.3211 - accuracy: 0.9283
loss :  0.3211069107055664
acc :  0.9283000230789185

learning_rate=0.001
시간 :  18.46 초
313/313 [==============================] - 1s 2ms/step - loss: 0.1805 - accuracy: 0.9540
loss :  0.18048140406608582
acc :  0.9539999961853027

learning_rate=0.0001
시간 :  48.36 초
313/313 [==============================] - 1s 2ms/step - loss: 0.1698 - accuracy: 0.9601
loss :  0.16975022852420807
acc :  0.960099995136261
'''