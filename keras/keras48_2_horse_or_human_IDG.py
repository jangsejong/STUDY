# 1.세이브
# 2.세이브한뒤에 주석처리
# 3.

# import os

# # horses/humans 데이터셋 경로 지정
# train_horse_dir = '../_data/image/horse-or-human/horses'
# train_human_dir = '../_data/image/horse-or-human/humans'

# # horses 파일 이름 리스트
# train_horse_names = os.listdir(train_horse_dir)
# print(train_horse_names[:10])

# # humans 파일 이름 리스트
# train_human_names = os.listdir(train_human_dir)
# print(train_human_names[:10])

# # horses/humans 총 이미지 파일 개수
# print('total training horse images:', len(os.listdir(train_horse_dir)))  #500
# print('total training human images:', len(os.listdir(train_human_dir)))  #527

import numpy as np
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
    '../_data/image/horse-or-human',
    target_size=(150,150),  #사이즈조절가능
    batch_size=600,
    class_mode='categorical',
    shuffle=True,
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human',
    target_size=(150,150),  #사이즈조절가능
    batch_size=600,
    class_mode='categorical',
    subset='validation') # set as validation data

     

print(train_generator[0][0].shape)     #(600, 150, 150, 3)
print(validation_generator[0][0].shape)     #(205, 150, 150, 3)

np.save('../_save_npy/keras48_2_1_train_x.npy', arr= train_generator[0][0])
np.save('../_save_npy/keras48_2_1_train_y.npy', arr= train_generator[0][1])
np.save('../_save_npy/keras48_2_1_test_x.npy', arr= validation_generator[0][0])
np.save('../_save_npy/keras48_2_1_test_y.npy', arr= validation_generator[0][1])

# print(train_generator[0])
# print(validation_generator[0])

# xy_train = train_datagen.flow_from_directory(         
#     '../_data/image/horse-or-human/training_set',
#     target_size = (50,50),                         
#     batch_size = 10,
#     class_mode = 'binary',
#     shuffle = True,
#     )           

# xy_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set',
#     target_size = (50,50),
#     batch_size = 10, 
#     class_mode = 'binary',
# )

# print(xy_train[0][0].shape, xy_train[0][1].shape)  # (10, 50, 50, 3) (10,)

# np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(150,150,3), activation='relu'))
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

hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 1, 
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


# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

'''
loss: 0.678576648235321
val_loss: 0.9945012331008911
acc: 0.5733333230018616
val_acc: 0.5121951103210449
'''