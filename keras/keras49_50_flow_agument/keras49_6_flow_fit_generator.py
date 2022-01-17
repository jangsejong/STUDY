from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5, 
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size)  #randint
# print(x_train.shape[0])                   # 60000
# print(randidx)                            # [28809 11925 51827 ... 34693  6672 48569]
# print(np.min(randidx), np.max(randidx))   # 0 59998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)                   # (40000, 28, 28)
# print(y_augmented.shape)                   # (40000, )  ?

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


xy_train = train_datagen.flow(x_augmented, y_augmented, #np.zeors(augment_size),
                                 batch_size=100, shuffle=False
                                 )#.next()
xy_test = test_datagen.flow(x_test, y_test, #np.zeors(augment_size),
                                 batch_size=100, shuffle=False
                                 )#.next()     

# print(xy_train)
# print(xy_train[0].shape,xy_train[1].shape) #(40000, 28, 28, 1) (40000,)

#모델
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#컴파일,훈련
hist = model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit_generator(xy_train, epochs=10,steps_per_epoch = len(xy_train))
                    # validation_data = validation_generator,
                    # validation_steps = 400,)


#평가
loss, acc = model.evaluate_generator(xy_test)
print("Accuracy : ", str(np.round(acc ,2)*100)+ "%")
#TypeError: 'float' object is not subscriptable

test_loss, test_acc = model.evaluate(xy_test)
print('loss :', test_loss)
print('acc :', test_acc)
                 
'''
Accuracy :  85.0%
loss : 0.3960881531238556
acc : 0.8532000184059143

'''                    
