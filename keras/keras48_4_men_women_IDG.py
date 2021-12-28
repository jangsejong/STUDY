import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

train_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women',
    target_size=(150,150),  #사이즈조절가능
    batch_size=2648,
    class_mode='categorical',
    shuffle=True,
    subset='training',seed=66,
    # color_mode='grayscale',
) # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women',
    target_size=(150,150),  #사이즈조절가능
    batch_size=661,
    class_mode='categorical',
    subset='validation',seed=66,
    # color_mode='grayscale',
) # set as validation data

# print('total train_generator images:', len(os.listdir(train_generator)))  #2648 
# print('total validation_generator images:', len(os.listdir(validation_generator)))  #661
   

print(train_generator[0][0].shape)     #(2648, 150, 150, 3)
print(validation_generator[0][0].shape)     #(661, 150, 150, 3)

np.save('../_save_npy/keras48_4_1_train_x.npy', arr= train_generator[0][0])
np.save('../_save_npy/keras48_4_1_train_y.npy', arr= train_generator[0][1])
np.save('../_save_npy/keras48_4_1_test_x.npy', arr= validation_generator[0][0])
np.save('../_save_npy/keras48_4_1_test_y.npy', arr= validation_generator[0][1])



# 2. 모델
from tensorflow.keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 4, 
                    validation_data = validation_generator,
                    validation_steps = 4,)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])
'''
loss: 0.7136183381080627
val_loss: 3.180493116378784
acc: 0.43957704305648804
val_acc: 0.5748865604400635


'''


