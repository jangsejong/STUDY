import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000 , 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000 , 28, 28, 1).astype('float32') / 255.

#2. model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D
from tensorflow.keras.models import Sequential, Model

# def autoencoder(hidden_layer_size):
#     input_shape = Input(shape=(784, ))
#     encoded = Dense(units=hidden_layer_size, activation='relu')(input_shape)    
#     decoded = Dense(units=784, activation='sigmoid')(encoded)    
#     autoencoder = Model(input_shape, decoded)    
#     return autoencoder

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, activation='relu', input_shape=(784,)))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model

# def autoencoder(hidden_layer_size):
#     model = Sequential([
#     Dense(units=hidden_layer_size, activation='relu', input_shape=(784,)),
#     Dense(units=784, activation='sigmoid'),
#     ])
#     return model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    return model


model = autoencoder(hidden_layer_size=128)

model.compile(optimizer='adam', loss='mse')
# model.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True, validation_split=0.2)
model.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)


#4. evaluate
output = model.predict(x_test)



