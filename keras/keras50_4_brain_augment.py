import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


#1. Data

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=False,
    vertical_flip=False,
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
    shuffle=True,
    subset='training'
)                                     #Found 160 images belong to 2 classes.


xy_test = train_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    subset='validation'
)                                   #Found 120 images belong to 2 classes.

print(xy_train[0][0].shape)   #(10, 150, 150, 3)
print(xy_test[0][0].shape)     #(10, 150, 150, 3)



augment_size = 160
randidx = np.random.randint(xy_train[0][0].shape[0], size = augment_size)  #randint
# 0부터 x_train.shape[0] 까지의 값을 argument_size 만큼 뽑겠다 -> 리스트형태
# print(x_train.shape[0])                   # 60000
# print(randidx)                            # [28809 11925 51827 ... 34693  6672 48569]
# print(np.min(randidx), np.max(randidx))   # 0 59998

x_argumented = xy_train[0][0][randidx].copy()
y_argumented = xy_train[0][1][randidx].copy()

# print(x_argumented.shape)                # (40000, 28, 28)
# print(y_augmented.shape)                   # (40000, )  ?

xy_train = np.concatenate((xy_train, xy_augmented))

'''
print(xy_train.shape)  #(100000, 28, 28, 1)              








x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)





#2. 모델
from tensorflow.keras.models import Sequential, Model, load_model, Model,save_model
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
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

mpdel = np.save("./_save_npy/keras47_5_save_model.h5")

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

loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy:', loss[1])

y_predict=model.predict(x_test)
print(y_predict.shape, y_test.shape)
print(y_predict[0])
print(y_test[0])
# print(y_test)
# print(y_test.shape, y_predict.shape)

y_predict=np.argmax(y_predict,axis=1)
# y_test=np.argmax(y_test,axis=1)
# print(y_predict.shape, y_test.shape)

from sklearn.metrics import r2_score, accuracy_score

a=accuracy_score(y_test, y_predict)
print('accuracy score:' , a)


'''

