from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, LSTM
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dropout



model = Sequential()
##model.add(Conv2D(10, (2, 2), strides=1, padding='valid', input_shape=(10, 10, 1), activation='relu')) 
model.add(Conv2D(128 ,kernel_size=(2,2), padding='vaild', input_shape=(28, 28, 1)))     
model.add(Dropout(0.2))                     
model.add(Conv2D(128,kernel_size=(2,2), padding='same', activation='relu'))  #     (None, 13, 13, 5)                          
model.add(MaxPooling2D())

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
# model.add(Dense(2, activation='linear'))
# model.add(Dense(1, activation='softmax'))

model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 5)         205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 7)         147
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 7)         203
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 10)        290
_________________________________________________________________
flatten (Flatten)            (None, 1000)              0
_________________________________________________________________
reshape (Reshape)            (None, 100, 10)           0
_________________________________________________________________
conv1d (Conv1D)              (None, 99, 5)             105
_________________________________________________________________
lstm (LSTM)                  (None, 15)                1260
_________________________________________________________________
dense (Dense)                (None, 10)                160
=================================================================
Total params: 2,420
Trainable params: 2,420
Non-trainable params: 0
_______________________________________________
'''



