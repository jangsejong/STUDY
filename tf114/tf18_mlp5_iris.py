from sklearn import datasets
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)   # (150, 4) (150,)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data.reshape(-1,1))
print(y_data.shape) # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)     # (30, 4) (30, 3)

# input_layer 
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([4,3]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1,3]), name = 'bias1')


#2. 모델구성
#h-1
hedden_layer1 = tf.nn.softmax(tf.matmul(x , w1) + b1)
# hedden_layer1 = tf.matmul(x , w1) + b1
# hedden_layer1 = tf.nn.relu(tf.sigmoid(tf.matmul(x , w1) + b1))
w2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([3,8]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1,8]), name = 'bias2')

#h-2
hedden_layer2 = tf.nn.softmax(tf.matmul(hedden_layer1, w2) + b2)
w3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([8,12]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1,12]), name = 'bias3')

#h-3
hedden_layer3 = tf.nn.softmax(tf.matmul(hedden_layer2, w3) + b3)
w4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([12,6]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1,6]), name = 'bias4')

#h-4
hedden_layer4 = tf.nn.softmax(tf.matmul(hedden_layer3, w4) + b4)
w5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([6,3]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1,3]), name = 'bias5')


output_layer = tf.nn.softmax(tf.matmul(hedden_layer4, w5) + b5)


#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output_layer), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(201):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_train, y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
    
    results = sess.run(output_layer, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1)))     
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_data, y:y_data})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()

# 예측결과 :  [1 1 1 0 1 1 0 0 0 1 2 2 0 2 2 0 1 1 1 2 0 1 1 2 1 2 0 0 1 2]
# accuracy :  0.0

