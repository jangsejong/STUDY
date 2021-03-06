import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Conv2D, Flatten
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
model.add(LSTM(200 , input_shape=(28, 28)))                          # (9, 9, 10)                             # (7, 7, 5)
#model.add(Conv2D(100,kernel_size=(2,2), activation='relu')) 
model.add(Flatten())                              
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')

#4. 예측, 결과
test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc :', test_acc)
'''
acc : 0.9788333177566528
----------------------
LSTM 반영시
걸린시간 :  126.511 초
acc : 0.9825000166893005
========================

'''