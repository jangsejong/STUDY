# 1.세이브
# 2.세이브한뒤에 주석처리
# 3.

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


