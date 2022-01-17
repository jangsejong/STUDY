from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Conv2D, Flatten, MaxPooling2D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler 
import time

#시작

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)  3072
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)


x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)


print(np.unique(y_train, return_counts=True))  #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train=x_train.reshape(50000,-1)   
# x_test=x_test.reshape(10000,-1)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 96, 32)
x_test = x_test.reshape(x_test.shape[0], 96, 32)


model=Sequential()
model.add(Conv1D(64,2, input_shape=(96,32)))
model.add(Flatten()) ##
model.add(Dense(64, activation='relu'))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

############################################################################################################
# import datetime
# date=datetime.datetime.now()   
# datetime = date.strftime("%m%d_%H%M")   #1206_0456
# #print(datetime)

# filepath='./_ModelCheckPoint/'
# filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
# model_path = "".join([filepath, 'k26_', datetime, '_', filename])
#             #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es=EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# mcp=ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1, 
#                     save_best_only=True,filepath=model_path)

start = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=160, validation_split=0.2)#, mcp])
# model.save('./_save/keras33_save_model.h5')
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')


loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
<CNN>
시간 :  4565.11 초
313/313 [==============================] - 2s 5ms/step - loss: 0.9039 - accuracy: 0.6799
loss :  0.9039157032966614
accuracy :  0.6798999905586243
<DNN>
Epoch 00837: val_loss did not improve from 2.30258
시간 :  1393.26 초
313/313 [==============================] - 0s 982us/step - loss: 2.3026 - accuracy: 0.1000
loss :  1.6646974086761475
accuracy :  0.40049999952316284
<LSTM>
걸린시간 :  255.381 초
loss :  2.0039901733398438
accuracy :  0.27090001106262207
<Conv1D>
걸린시간 :  30.221 초
loss :  2.305962085723877
accuracy :  0.10000000149011612
'''