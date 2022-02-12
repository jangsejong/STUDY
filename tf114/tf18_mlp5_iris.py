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
# #1
# x_train = x_data[:120]
# x_test = x_data[120:]
# y_test = y_data[120:]
# y_train = y_data[:120]

# #2
# x_train = x_data.split.train.subsplit(0.8, shuffle=True, random_state=66)
# x_test = x_data.split.test.subsplit(0.2, shuffle=True, random_state=66)
# y_train = y_data.split.train.subsplit(0.8, shuffle=True, random_state=66)
# y_test = y_data.split.test.subsplit(0.2, shuffle=True, random_state=66)

#3
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

# #4
# train_dataset = datasets.load('iris', split='train[:20%]', as_supervised=True)
# test_dataset = datasets.load('iris', split='train[20%:]', as_supervised=True)
# train_x, train_y = train_dataset.data, train_dataset.target
# test_x, test_y = test_dataset.data, test_dataset.target
# print(train_x.shape, train_y.shape) # (120, 4) (120,)
# print(test_x.shape, test_y.shape) # (30, 4) (30,)
# print(train_x[0], train_y[0]) # [5.1 3.5 1.4 0.2] 0
# print(test_x[0], test_y[0]) 

print(x_train.shape, y_train.shape)   # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)     # (30, 4) (30, 3)


#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])
w1 = tf.compat.v1.Variable(tf.random.normal([4,100]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([1,100]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([100,10]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([1,10]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([10,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([1,10]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([10,3]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1,3]), name='bias4')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer3, w4)+b4)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # cross entropy
# loss = -tf.reduce_mean(y * tf.math.log(hypothesis)) # cross entropy
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000000001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) # Adam

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(21):
        _, cost_val, hy_val = sess.run([optimizer, loss, hypothesis], feed_dict={x:x_train, y:y_train})
        if step % 10 ==0:
            print(step, cost_val)
    
    
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1)))     
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_data, y:y_data})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()