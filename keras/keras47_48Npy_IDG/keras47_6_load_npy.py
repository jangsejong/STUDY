import numpy as np
from numpy.core.defchararray import array
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


#1. Data

train_datagen = ImageDataGenerator(
    # rescale = 1./255,
    # horizontal_flip=False,
    # vertical_flip=False,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.1,
    # shear_range=0.9,
    # fill_mode='nearest',
    
    # featurewise_center=True,
    # featurewise_std_normalization=True,  
    # validation_split=0.1
)

test_datagen = ImageDataGenerator(                  #평가만 해야하기 때문에 증폭할 필요성이 없다.
    rescale=1./255
)                                   #Found 160 images belong to 2 classes.

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(150,150),  #사이즈조절가능
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

xy_test = train_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),  #사이즈조절가능
    batch_size=200,
    class_mode='binary'
)                                   #Found 120 images belong to 2 classes.

print(xy_train[0][0].shape)     #(160, 150, 150, 3) 
print(xy_train[0][1].shape)     #(160,)
print(xy_test[0][0].shape)     #(120, 150, 150, 3)
print(xy_test[0][1].shape)     #(120,)

# np.save('./_save_npy/keras47_5_train_x.npy', arr= xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr= xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr= xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr= xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')
print(x_train)
print(x_train.shape)
