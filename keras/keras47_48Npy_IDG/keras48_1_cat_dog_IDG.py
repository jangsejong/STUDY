# 1.세이브
# 2.세이브한뒤에 주석처리
# 3.

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

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
    validation_split=0.2
)

test_datagen = ImageDataGenerator(                  #평가만 해야하기 때문에 증폭할 필요성이 없다.
    rescale=1./255
)                                   #Found 160 images belong to 2 classes.

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set',
    target_size=(15,15),  #사이즈조절가능
    batch_size=200,
    class_mode='categorical',
    shuffle=True
)

xy_test = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set',
    target_size=(15,15),  #사이즈조절가능
    batch_size=200,
    class_mode='categorical'
)                                   #Found 120 images belong to 2 classes.
     

print(xy_train[0][0].shape)     #(200, 15, 15, 3)
print(xy_train[0][1].shape)     #(200, 1)

np.save('../_save_npy/keras48_1_train_x.npy', arr= xy_train[0][0])
np.save('../_save_npy/keras48_1_train_y.npy', arr= xy_train[0][1])
np.save('../_save_npy/keras48_1_test_x.npy', arr= xy_test[0][0])
np.save('../_save_npy/keras48_1_test_y.npy', arr= xy_test[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(15,15,3), activation='relu'))
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
model.add(Dense(2, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs = 20, steps_per_epoch = 1, 
                    validation_data = test_datagen,
                    validation_steps = 4,)


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

