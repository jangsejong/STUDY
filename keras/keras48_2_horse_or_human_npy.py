# 1.세이브
# 2.세이브한뒤에 주석처리
# 3.
import numpy as np
import os

# horses/humans 데이터셋 경로 지정
train_horse_dir = '../_data/image/horse-or-human/horses'
train_human_dir = '../_data/image/horse-or-human/humans'

# horses 파일 이름 리스트
train_horse_names = os.listdir(train_horse_dir)
# print(train_horse_names[:10])

# humans 파일 이름 리스트
train_human_names = os.listdir(train_human_dir)
# print(train_human_names[:10])

# horses/humans 총 이미지 파일 개수
# print('total training horse images:', len(os.listdir(train_horse_dir)))  #500
# print('total training human images:', len(os.listdir(train_human_dir)))  #527

#이미지확인하기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#모델구성하기

from tensorflow.keras.models import Sequential
from keras.layers import *
import tensorflow as tf

# model = tf.keras.models.Sequential([
#     # The first convolution
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
#     tf.keras.layers.MaxPool2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     # The third convolution
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     # The fourth convolution
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     # The fifth convolution
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     # Flatten
#     tf.keras.layers.Flatten(),
#     # 512 Neuron (Hidden layer)
#     tf.keras.layers.Dense(512, activation='relu'),
#     # 1 Output neuron
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.summary()


model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(300, 300, 3), activation='relu'))
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




#모델 컴파일

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy'])


#이미지 데이터 전처리

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
  '../_data/image/horse-or-human',
  target_size=(300, 300),
  batch_size=128,
  class_mode='categorical'
)

#모델 훈련하기
hist = model.fit(train_generator,steps_per_epoch=8,epochs=15,verbose=1)

# hist = model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])#, mcp]) ) 

model.save("./_save_npy/keras48_2_save_weights.h5")

import matplotlib.pyplot as plt


accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
# val_loss = hist.history['val_loss']
epochs = range(len(accuracy))

print('acc : ', accuracy[-1])
#print('val_acc : ', val_accuracy[-1])
print('loss : ', loss[-1])
#print('val_loss : ', val_loss[-1])

# plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
# #plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'go', label='Training Loss')
# #plt.plot(epochs, val_loss, 'g', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()
#print("걸린시간 : ", round(end, 3), '초')
'''
acc :  0.9988876581192017
loss :  0.010299109853804111
'''