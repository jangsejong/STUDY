import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.keras.applications import VGG16 # 기존 모델을 사용하기 위해 임포트

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

vgg16.trainable = False # weight를 사용하지 않겠다.
vgg16.trainable = True # weight를 전부 불러오게 하기 위해 trainable을 True로 설정   

model = Sequential
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


model.summary()

