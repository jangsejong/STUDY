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

def autoencoder1(hidden1, hidden2, hidden3, hidden4, hidden5):
    model = Sequential()
    model.add(Dense(units=hidden1, activation='relu', input_shape=(784,)))
    model.add(Dense(units=hidden2, activation='relu'))
    model.add(Dense(units=hidden3, activation='relu'))
    model.add(Dense(units=hidden4, activation='relu'))
    model.add(Dense(units=hidden5, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model



model = autoencoder1(16, 64, 128, 256, 512)

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True, validation_split=0.2)


#4. evaluate
output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다.
for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='Greys_r')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='Greys_r')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()