import numpy as np
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.metrics import accuracy


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape, y_train.shape)           #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)           #(10000, 28, 28) (10000,)
x_train, x_test = x_train/255.0, x_test/255.0

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_test.shape)                #(10000, 28, 28, 1)  
#print(np.unique(y_train,return_counts=True)) 
x = x_train

from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델 구성
model = Sequential()
##model.add(Conv2D(10, (2, 2), padding='valid', input_shape=(10, 10, 1), activation='relu')) # (9, 9, 10)
model.add(Conv2D(200 ,kernel_size=(2,2), input_shape=(28, 28, 1)))                          # (9, 9, 10)                             # (7, 7, 5)
model.add(Conv2D(100,kernel_size=(2,2), activation='relu')) 
model.add(Flatten())                              
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 5 , mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과
test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc :', test_acc)


#acc : 0.9788333177566528