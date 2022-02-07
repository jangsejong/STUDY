import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Conv2D, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape)           #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)           #(10000, 28, 28) (10000,)
x_train, x_test = x_train/255.0, x_test/255.0

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)                #(10000, 28, 28, 1)  

#print(np.unique(y_train,return_counts=True))   #라벨갯수파악필요, 성능차이가 있다

#평가지표acc (0.98 이상) 이벨류에이트테스트, 발리데이션테스트,메트릭스에큐러시

x = x_train

from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

#2. 모델 구성
model = Sequential()
##model.add(Conv2D(10, (2, 2), padding='valid', input_shape=(10, 10, 1), activation='relu')) # (9, 9, 10)
model.add(Conv1D(80,2 , input_shape=(28, 28)))                          # (9, 9, 10)                             # (7, 7, 5)
#model.add(Conv2D(100,kernel_size=(2,2), activation='relu')) 
model.add(Flatten())                              
model.add(Dense(40, activation='relu'))
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

##################시각화###########################
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

#1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'],maker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],maker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'],maker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'],maker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


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