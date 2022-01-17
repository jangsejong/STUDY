import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

x_train = np.load('../_save_npy/keras48_3_1_train_x.npy')
y_train = np.load('../_save_npy/keras48_3_1_train_y.npy')
x_test = np.load('../_save_npy/keras48_3_1_test_x.npy')
y_test = np.load('../_save_npy/keras48_3_1_test_y.npy')



#2. 모델
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
model.add(Dense(3, activation='softmax'))


#model.summary()


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'auto', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras48_1_MCP.hdf5')

start = time.time()
             
# hist = model.fit(x_train, y_train, epochs=50,
                    
#                      callbacks = [es])#, mcp]) )                  #과제
#                     # 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 
#                     # 총 160 개의 검증 샘플이 있고 배치사이즈가 5이므로 4의 배수스텝으로 지정합니다.
hist = model.fit(x_train, y_train, epochs=500, batch_size=2, verbose=1, validation_split=0.2, callbacks=[es])#, mcp]) ) 
                    
end = time.time()- start

# model.fit(xy_train[0][0],xy_train[0][1])

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("걸린시간 : ", round(end, 3), '초')
model.save("./_save_npy/keras48_3_save_npy.h5")

# 그래프 그리기
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print('acc : ', accuracy[-1])
print('val_acc : ', val_accuracy[-1])
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])

'''
acc :  0.9478908181190491
val_acc :  0.8118811845779419
loss :  0.18983221054077148
val_loss :  0.7259063124656677
'''