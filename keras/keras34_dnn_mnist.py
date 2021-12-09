from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#실습!!!

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()




#2. 모델구성
model = Sequential()
#model.add(Dense(64, input_shape=(28*28, )))
model.add(Dense(64, input_shape=(784, )))

