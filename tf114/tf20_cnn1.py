from pickletools import optimize
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

tf.compat.v1.set_random_seed(66)

# 1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])

#2. 모델

w1 = tf.get_variable("w1", shape=[2, 2, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
#<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
b = tf.Variable(tf.random_normal([64]))

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#<tf.Tensor 'Conv2D_1:0' shape=(?, 28, 28, 64) dtype=float32>
print(L1)

