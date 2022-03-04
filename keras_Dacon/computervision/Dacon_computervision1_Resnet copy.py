import tensorflow as tf 
tf.config.list_logical_devices()
import os
import warnings
warnings.filterwarnings(action='ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU 할당

#create training dataset
from glob import glob
import numpy as np
import PIL
from PIL import Image

path = 'D:/Study/_data/dacon/vision/train/'

training_images = []
training_labels = []

for filename in glob(path +"*"):
    for img in glob(filename + "/*.jpg"):
        an_img = PIL.Image.open(img) #read img
        img_array = np.array(an_img) #img to array
        training_images.append(img_array) #append array to training_images
        label = filename.split('\\')[-1] #get label
        training_labels.append(label) #append label
        
training_images = np.array(training_images)
training_labels = np.array(training_labels)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
training_labels= le.fit_transform(training_labels)
training_labels = training_labels.reshape(-1,1)

print(training_images.shape)
print(training_labels.shape)

#create test dataset

path = 'D:/Study/_data/dacon/vision/test/'

test_images = []
test_idx = []

flist = sorted(glob(path + '*.jpg'))

for filename in flist:
    an_img = PIL.Image.open(filename) #read img
    img_array = np.array(an_img) #img to array
    test_images.append(img_array) #append array to training_images 
    label = filename.split('\\')[-1] #get id 
    test_idx.append(label) #append id
    
test_images = np.array(test_images)

print(test_images.shape)
print(test_idx[0:5])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.random.set_seed(42)

image_generator = ImageDataGenerator(
    rotation_range=20,
    brightness_range = [0.6, 1.0],
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

training_image_aug = image_generator.flow(training_images, np.zeros(50000), batch_size=50000, shuffle=False, seed = 42).next()[0]
training_image_aug_2 = image_generator.flow(training_images, np.zeros(50000), batch_size=50000, shuffle=False, seed = 42^2).next()[0]
training_image_aug_3 = image_generator.flow(training_images, np.zeros(50000), batch_size=50000, shuffle=False, seed = 42^3).next()[0]
training_image_aug_4 = image_generator.flow(training_images, np.zeros(50000), batch_size=50000, shuffle=False, seed = 42^4).next()[0]

training_images = np.concatenate((training_images, 
                                  training_image_aug, 
                                  training_image_aug_2, 
                                  training_image_aug_3, 
                                  training_image_aug_4))

training_labels = np.concatenate((training_labels, 
                                  training_labels, 
                                  training_labels, 
                                  training_labels, 
                                  training_labels))

training_labels = tf.one_hot(training_labels, 10) #one-hot 기법 적용
training_labels = np.array(training_labels)
training_labels = training_labels.reshape(-1,10) #one-hot 기법을 적용했다면, shape을 바꿔줍니다.

print(training_images.shape)
print(training_labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(training_images, 
                                                      training_labels, 
                                                      test_size=0.1, 
                                                      stratify = training_labels, 
                                                      random_state=66,
                                                      shuffle = True)

X_test = test_images

print('X_train 크기:',X_train.shape)
print('y_train 크기:',y_train.shape)
print('X_valid 크기:',X_valid.shape)
print('y_valid 크기:',y_valid.shape)
print('X_test  크기:',X_test.shape)

X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

def identity_block(X, filters, kernel_size):
    X_shortcut = X
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    
    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def convolutional_block(X, filters, kernel_size):
    X_shortcut = X
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X_shortcut = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)
    
    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def ResNet50CL(input_shape = (32, 32, 3), classes = 10):
    X_input = tf.keras.layers.Input(input_shape)
    X = X_input
    
    X = convolutional_block(X, 64, (3,3)) #conv
    X = identity_block(X, 64, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 128, (3,3)) #64->128, use conv block
    X = identity_block(X, 128, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 256, (3,3)) #128->256, use conv block
    X = identity_block(X, 256, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 512, (3,3)) #256->512, use conv block
    X = identity_block(X, 512, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(10, activation = 'softmax')(X) # ouput layer (10 class)

    model = tf.keras.models.Model(inputs = X_input, outputs = X, name = "ResNet50CL")
    
    return model

def ResNet50C(input_shape = (32, 32, 3), classes = 10):
    X_input = tf.keras.layers.Input(input_shape)
    X = X_input
    
    X = tf.keras.layers.Conv2D(64, (3,3), padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = convolutional_block(X, 64, (3,3)) #use conv block (?)
    X = identity_block(X, 64, (3,3))
    X = identity_block(X, 64, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 128, (3,3)) #64->128, use conv block
    X = identity_block(X, 128, (3,3))
    X = identity_block(X, 128, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 256, (3,3)) #128->256, use conv block
    X = identity_block(X, 256, (3,3))
    X = identity_block(X, 256, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 512, (3,3)) #256->512, use conv block
    X = identity_block(X, 512, (3,3))
    X = identity_block(X, 512, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(10, activation = 'softmax')(X) # ouput layer (10 class)

    model = tf.keras.models.Model(inputs = X_input, outputs = X, name = "ResNet50C")
    
    return model

model = ResNet50CL()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

EPOCH = 85
BATCH_SIZE = 135

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                              patience=3, 
                             )

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=4,
)

data = model.fit(X_train, 
                 y_train, 
                 validation_data=(X_valid, y_valid), 
                 epochs=EPOCH, 
                 batch_size=BATCH_SIZE, 
                 callbacks=[reduceLR, earlystopping],)

import matplotlib.pyplot as plot

plot.plot(data.history['accuracy'])
plot.plot(data.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
# plot.show()

plot.plot(data.history['loss'])
plot.plot(data.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
# plot.show()

model.save('ResNet50CL.h5')

model = ResNet50C()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

EPOCH = 85
BATCH_SIZE = 135

filename = 'resnet50C-checkpoint.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filename,             # file명을 지정합니다
                                                monitor='val_accuracy',   # val_accuracy 값이 개선되었을때 호출됩니다
                                                verbose=1,            # 로그를 출력합니다
                                                save_best_only=True,  # 가장 best 값만 저장합니다
                                                mode='auto'           # auto는 알아서 best를 찾습니다. min/max (loss->min, accuracy->max)
                                               )

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', #stop 조건으로 관찰할 변수 선택
                                                 patience=10,            #10 Epoch동안 (val-accuracy가)개선되지 않는다면 종료
                                                )

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', #lr을 낮출 조건으로 관찰할 변수 선택
                                                factor=0.5,             #조건이 충족되었을때 LR에 factor를 곱함 (2분의 1배가 됨)
                                                patience=6,             #10 Epoch동안 (val-accuracy가)개선되지 않는다면 lr 감소
                                               )

data = model.fit(X_train, 
                 y_train, 
                 validation_data=(X_valid, y_valid), 
                 epochs=EPOCH, 
                 batch_size=BATCH_SIZE, 
                 callbacks=[reduceLR, earlystopping, checkpoint],)

import matplotlib.pyplot as plot

plot.plot(data.history['accuracy'])
plot.plot(data.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
# plot.show()

plot.plot(data.history['loss'])
plot.plot(data.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
# plot.show()

model = tf.keras.models.load_model('./resnet50CL.h5') #학습했던 Resnet50CL 불러오기

X_test_ori = test_images
X_test = test_images
X_test = X_test / 255.0

pred_proba = model.predict(X_test)

#TTA 적용
for i in [1, 2, 3, 4]:
    X_test_aug = image_generator.flow(X_test_ori, np.zeros(10000), batch_size=10000, shuffle=False, seed = 66^i).next()[0]
    X_test_aug = X_test_aug / 255.0
    pred_proba_aug = model.predict(X_test_aug)
    pred_proba = np.add(pred_proba, pred_proba_aug)
    
pred_class = []

for i in pred_proba:
    pred = np.argmax(i)
    pred_class.append(pred)
    
pred_class = le.inverse_transform(pred_class)
pred_class[0:5]

model = tf.keras.models.load_model('./resnet50C-checkpoint.h5') #학습했던 Resnet50C 불러오기

pred_proba_2 = model.predict(X_test)
pred_proba = np.add(pred_proba, pred_proba_2) #resnet50CL 결과에 추론결과를 계속 더함

#TTA 적용
for i in [1, 2, 3, 4]:
    X_test_aug = image_generator.flow(X_test_ori, np.zeros(10000), batch_size=10000, shuffle=False, seed = 42^i).next()[0]
    X_test_aug = X_test_aug / 255.0
    pred_proba_aug = model.predict(X_test_aug)
    pred_proba = np.add(pred_proba, pred_proba_aug)
    
pred_class = []

for i in pred_proba:
    pred = np.argmax(i)
    pred_class.append(pred)
    
pred_class = le.inverse_transform(pred_class)
pred_class[0:5]

import pandas as pd

sample_submission = pd.read_csv("D:/Study/_data/dacon/vision/sample_submission.csv")

sample_submission.target = pred_class
sample_submission.to_csv("D:/Study/_data/dacon/vision/0304_1.csv",index=False)

sample_submission.head(10)

model_imp = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=(32,32,3))
model_imp.summary()
