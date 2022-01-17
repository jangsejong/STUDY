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


dim = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
out_node = len(np.unique(y_train))

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(300, activation='relu', input_dim = dim))
model.add(MaxPooling2D())                   
model.add(Flatten())                             
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 15 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과

test_loss, test_acc = model.evaluate(x_test, y_test)
print('loss : ', test_loss)
print('acc :', test_acc)

'''
loss :  2.8910043239593506
acc : 0.31450000405311584

after scaling
loss :  2.794773578643799
acc : 0.33820000290870667
'''