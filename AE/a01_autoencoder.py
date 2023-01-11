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

input_shape = Input(shape=(784, ))

encoded = Dense(units=154, activation='relu')(input_shape)
# encoded = Dense(units=256, activation='relu')(input_shape)

decoded = Dense(units=784, activation='sigmoid')(encoded)

autoencoder = Model(input_shape, decoded)

#3. traininig
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#4. fit
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, shuffle=True, validation_split=0.2)

#5. evaluate

decoded_imgs = autoencoder.predict(x_test)

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
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
