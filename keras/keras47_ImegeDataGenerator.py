import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,  
    validation_split=0.2
)

test_datagen = ImageDataGenerator(                  #평가할때는 원래 이미지로 평가해야하기때문에 변형하지 않는다.
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(100,100),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
