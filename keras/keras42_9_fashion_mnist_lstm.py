import numpy as np
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x = x_train

from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape) # (60000, 10)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

#2. 모델 구성
model = Sequential()
model.add(LSTM(200 , activation='relu', input_shape=(28, 28)))                      
model.add(Flatten())                              
model.add(Dense(40))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과
test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc :', test_acc)

'''
#acc : 0.8949999809265137

LSTM 반영시
acc : 0.8992499709129333
'''