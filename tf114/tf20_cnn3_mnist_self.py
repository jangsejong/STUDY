from pickletools import optimize
from unittest import result
from sklearn.metrics import accuracy_score
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

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델
#layer1
w1 = tf.get_variable("w1", shape=[2, 2, 1, 32]) # 
#<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#<tf.Tensor 'Conv2D_1:0' shape=(?, 28, 28, 64) dtype=float32>
#print(L1)
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L1_maxpool) # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

#layer2
w2 = tf.get_variable("w2", shape=[3, 3, 32, 16])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME') 
# Tensor("Conv2D_2:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L2_maxpool) # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

#layer3
w3 = tf.get_variable("w3", shape=[3, 3, 16, 8])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#layer4
w4 = tf.get_variable("w4", shape=[3, 3, 8, 16])
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L4_maxpool)

#Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*16])
print(L_flat) # Tensor("Reshape:0", shape=(?, 128), dtype=float32)

#layer5
L5 = tf.layers.dense(L_flat, 64, activation=tf.nn.relu)
L5 = tf.layers.dropout(L5, rate=0.4)

#layer6
L6 = tf.layers.dense(L5, 32, activation=tf.nn.relu)
L6 = tf.layers.dropout(L6, rate=0.4)

#layer7
L7 = tf.layers.dense(L6, 16, activation=tf.nn.relu)

#layer8
L8 = tf.layers.dense(L7, 10, activation=tf.nn.softmax)

#3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(L8), axis=1))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#4. 세션
with tf.Session() as sess: # with문을 사용하여 세션을 열어준다.
      sess.run(tf.compat.v1.global_variables_initializer())
      
      for step in range(201):
         _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
         if step % 200 ==0:
            print(step, loss_val) 
            
      results = sess.run(L8, feed_dict={x:x_test})
      print(results, sess.run(tf.math.argmax(results, axis=1)))
      
      accuracy = sess.run(tf.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(results, axis=1), tf.math.argmax(y_test, axis=1)), dtype=tf.float32)))
      pred = sess.run(tf.math.argmax(results, axis=1))
      print("예측결과 : ", pred)
      print("accuracy:", accuracy)
      
      sess.close()