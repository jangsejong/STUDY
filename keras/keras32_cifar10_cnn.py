import numpy as np
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy
import pandas as pd

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train, x_test = x_train/255.0, x_test/255.0

#x_train = x_train.reshape    #(50000, 32, 32, 3) (50000, 1)
#x_test = x_test.reshape      #(10000, 32, 32, 3) (10000, 1)
# print(x_train.shape, y_train.shape)          #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)            #(10000, 32, 32, 3) (10000, 1)



# x = x_train
# y= y_train
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
#print(y_train.shape) # (60000, 10)


#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
#scaler.fit(x_train) # 비율을 가져옴

#x_train = scaler.transform(x_train)  # 스케일러 비율이 적용되서 0~1.0 사이로 값이 다 바뀜 
#x_test = scaler.transform(x_test) 

#x_train = x_train.reshape(50000, 32,32,3)
#x_test = x_test.reshape(10000, 32,32,3)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(100 ,kernel_size=(2,2),strides=2, padding='valid', activation='relu', input_shape=(32, 32, 3))) 
model.add(MaxPooling2D())   
model.add(Conv2D(100,kernel_size=(2,2), activation='relu')) 
model.add(MaxPooling2D())           
model.add(Flatten())                              
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=30, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과
test_loss, test_acc = model.evaluate(x_test, y_test)
print('loss : ', test_loss)
print('acc :', test_acc)

'''
loss :  1.1138389110565186
acc : 0.6204000115394592

after scaling

loss :  1.0125701427459717
acc : 0.6586999893188477
'''