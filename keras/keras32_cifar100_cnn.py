import numpy as np
from tensorflow.keras.datasets import cifar100 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy
import pandas as pd

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

#x_train = x_train.reshape    #(50000, 32, 32, 3) (50000, 1)
#x_test = x_test.reshape      #(10000, 32, 32, 3) (10000, 1)
print(x_train.shape, y_train.shape)          #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)            #(10000, 32, 32, 3) (10000, 1)



x = x_train

from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
print(y_train.shape) # (60000, 10)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(200 ,kernel_size=(2,2),strides=2, padding='valid', activation='relu', input_shape=(32, 32, 3))) 
model.add(MaxPooling2D())                     
model.add(Flatten())                              
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과

test_loss, test_acc = model.evaluate(x_test, y_test)
print('loss : ', test_loss)
print('acc :', test_acc)

'''
loss :  2.878316879272461
acc : 0.31209999322891235
'''