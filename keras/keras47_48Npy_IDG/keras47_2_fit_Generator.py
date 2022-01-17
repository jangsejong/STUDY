import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


#1. Data

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest',
    
    featurewise_center=True,
    featurewise_std_normalization=True,  
    validation_split=0.1
)

test_datagen = ImageDataGenerator(                  #평가만 해야하기 때문에 증폭할 필요성이 없다.
    rescale=1./255
)                                   #Found 160 images belong to 2 classes.

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

xy_test = train_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary'
)                                   #Found 120 images belong to 2 classes.



#<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000125B62D4F70>


#print(xy_train[15])             # batch_size 영향을 받는다.

# x = datasets.data 
# y = datasets.target
# print(xy_train[31]) last batch
# print(xy_train[0][0])        
# print(xy_train[0][1])          
# print(xy_train[0][2])          

# print(xy_train[0][0].shape)     #(5, 150, 150, 3)   
# print(xy_train[0][1].shape)     #(5,)

# print(type(xy_train))               <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))            <class 'tuple'>
# print(type(xy_train[0][0]))         <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))         <class 'numpy.ndarray'>



#2. 모델
from tensorflow.keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3), padding='same', input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


#model.summary()


model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 5 , mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()
             
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,
                    validation_data=xy_test,
                    validation_steps=4 )                  #과제
                    # 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 
                    # 총 160 개의 검증 샘플이 있고 배치사이즈가 5이므로 4의 배수스텝으로 지정합니다.
end = time.time()- start

# model.fit(xy_train[0][0],xy_train[0][1])

accuracy = hist.history['accuracy']
val_accuracy = hist.history['accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("걸린시간 : ", round(end, 3), '초')

# 그래프 그리기
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()

print('acc : ', accuracy[-1])
print('val_acc : ', val_accuracy[-1])
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])



