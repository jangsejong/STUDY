import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000 , 784).astype('float32') / 255.
x_test = x_test.reshape(10000 , 784).astype('float32') / 255.

#2. model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
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

def autoencoder(hidden_layer_size):
    model = Sequential([
    Dense(units=hidden_layer_size, activation='relu', input_shape=(784,)),
    Dense(units=784, activation='sigmoid'),
    ])
    return model

model = autoencoder(hidden_layer_size=32)

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True, validation_split=0.2)


#4. evaluate
out_put = model.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(out_put[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()