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
'''
rotation_range: 이미지 회전 범위 (degrees)
width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
shear_range: 임의 전단 변환 (shearing transformation) 범위
zoom_range: 임의 확대/축소 범위
horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''
test_datagen = ImageDataGenerator(                  #평가만 해야하기 때문에 증폭할 필요성이 없다.
    rescale=1./255
)                                   #Found 160 images belong to 2 classes.

# D:\_data\image\brain

train_generator = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    subset='validation'
)                                   #Found 120 images belong to 2 classes.

print(train_generator[0][0].shape)  # 719
print(validation_generator[0][0].shape) # 308

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
from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


model.summary()

# 컴파일,훈련
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
             
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=200,
    # validation_data=testGenSet,
    validation_steps=10,
)