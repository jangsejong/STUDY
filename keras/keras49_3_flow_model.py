from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Conv2D, Flatten, MaxPooling2D, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.metrics import accuracy
import pandas as pd
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5, 
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size)  #randint
# print(x_train.shape[0])                   # 60000
# print(randidx)                            # [28809 11925 51827 ... 34693  6672 48569]
# print(np.min(randidx), np.max(randidx))   # 0 59998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)                   # (40000, 28, 28)
# print(y_augmented.shape)                   # (40000, )  ?

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeors(augment_size),
                                 batch_size=augment_size, shuffle=False
                                 ).next()[0]
# print(x_augmented)
# print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#print(x_train)         
# print(x_train.shape,y_train.shape)       #(100000, 28, 28, 1) (100000,)     




x_train, x_test = x_train/255.0, x_test/255.0

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

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
# model = Sequential()
# model.add(Conv1D(200 ,2 , activation='relu', input_shape=(28, 28)))                      
# model.add(Flatten())                              
# model.add(Dense(40))
# model.add(Dense(10, activation='softmax'))

from tensorflow.keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', patience= 5 , mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()


model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')


#4. 예측, 결과
test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc :', test_acc)

loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :

y_predict = model.predict(x_test)


R2 = r2_score(y_test, y_predict)
print(R2) # 

'''
#acc : 0.8949999809265137
--------------------------
LSTM 반영시
걸린시간 :  172.221 초
acc : 0.8789166808128357
==========================
Conv1D
걸린시간 :  20.374 초
acc : 0.8841666579246521
==========================
flow 증폭시
걸린시간 :  22.387 초
acc : 0.8072500228881836

ImageDataGenerator flip 주석
acc : 0.8343999981880188
'''