#훈련데이터 10만개로 증폭
#완료후 기존모델과 비교
# save_dir도 _temp 에 넣고
#중복데이터는 temp 에 저장후 훈련 끝난후 결과 보고 삭제


from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)  
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

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

augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size = augment_size)  #randint
# print(x_train.shape[0])                   # 50000
# print(randidx)                            # [28809 11925 51827 ... 34693  6672 48569]
# print(np.min(randidx), np.max(randidx))   # 0 49998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)                   # (50000, 32, 32, 3)
print(y_augmented.shape)                   # (50000, 1)


x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],x_augmented.shape[3])
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)
print(x_augmented.shape) #(50000, 32, 32, 3) 



x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape)  #(100000, 32, 32, 3)            
print(y_train.shape)  #(100000, 1)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# #scaler = MaxAbsScaler()

#모델
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))


#컴파일,훈련
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit(x_train,y_train, epochs=1, batch_size=256)
                    # validation_data = validation_generator,
                    # validation_steps = 400,)



#평가

# loss, acc = model.evaluate(x_test,y_test)
# print("Accuracy : ", str(np.round(acc ,2)*100)+ "%")
# #TypeError: 'float' object is not subscriptable

# test_loss, test_acc = model.evaluate(x_test,y_test)
# print('loss :', test_loss)
# print('acc :', test_acc)

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
loss :  2.8910043239593506
acc : 0.31450000405311584

after scaling
loss :  2.794773578643799
acc : 0.33820000290870667

after LSTM
걸린시간 :  1416.76 초
loss :  nan
acc : 0.009999999776482582
============================
Conv2D
걸린시간 :  98.129 초
loss :  4.3490800857543945
acc : 0.2542000114917755
============================
loss :  3.3614797592163086
accuracy: 0.19470000267028809
accuracy score: 0.1947
'''

                   
                 
