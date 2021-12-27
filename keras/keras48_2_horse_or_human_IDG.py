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

traintest_horse = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/horses',
    target_size=(150,150),  #사이즈조절가능
    batch_size=600,
    class_mode='binary',
    shuffle=True
)

traintest_human = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/humans',
    target_size=(150,150),  #사이즈조절가능
    batch_size=600,
    class_mode='binary'
)                                   #Found 120 images belong to 2 classes.
     

# print(train_horse.shape)     #(5, 150, 150, 3)
# print(train_human.shape)     #(5,)

# np.save('./_save_npy/keras48_2_train_x.npy', arr= train_horse_dir[0][0])
# np.save('./_save_npy/keras48_2_train_y.npy', arr= train_horse_dir[0][1])
# np.save('./_save_npy/keras48_2_test_x.npy', arr= train_human_dir[0][0])
# np.save('./_save_npy/keras48_2_test_y.npy', arr= train_human_dir[0][1])

