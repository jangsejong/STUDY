import datasets
from sklearn.datasets import load_diabetes
from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터

datasets = load_diabetes()

x_data = datasets.data # 열로 저장된 데이터
y_data = datasets.target # 실제값
print(x_data.shape) # (442, 10)
print(y_data.shape) # (442,)
y_data = y_data.reshape(-1,1) # (442,1)
print(y_data.shape) # (442,1)

# x_data = tf.expand_dims(x_data, axis=1)
# y_data = tf.expand_dims(y_data, axis=1)

x_train = x_data[:-100]
y_train = y_data[:-100]
x_test = x_data[-100:]
y_test = y_data[-100:]

INPUT_SIZE = 10
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 8
CLASSES = 1
Learning_Rate = 0.0007

x = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, CLASSES])
feed_dict = {x: x_train, y_: y_train}

w_h1 = tf.Variable(tf.random.normal([INPUT_SIZE, HIDDEN1_SIZE]))
b_h1 = tf.Variable(tf.random.normal([HIDDEN1_SIZE]))

w_h2 = tf.Variable(tf.random.normal([HIDDEN1_SIZE, HIDDEN2_SIZE]))
b_h2 = tf.Variable(tf.random.normal([HIDDEN2_SIZE]))

w_o = tf.Variable(tf.random.normal([HIDDEN2_SIZE, CLASSES]))
b_o = tf.Variable(tf.random.normal([CLASSES]))

param_list = [w_h1, b_h1, w_h2, b_h2, w_o, b_o]
# saver = tf.train.Saver(param_list)

H1 = tf.nn.relu(tf.matmul(x, w_h1) + b_h1)
H2 = tf.nn.relu(tf.matmul(H1, w_h2) + b_h2)
y = tf.matmul(H2, w_o) + b_o

cost = tf.reduce_mean(tf.square(y - y_))
train = tf.compat.v1.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

comp_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
r2 = tf.reduce_mean(tf.cast(comp_pred, tf.float32))

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={x: x_train, y_: y_train})
    if i % 100 == 0:
        print(i, sess.run(cost, feed_dict={x: x_train, y_: y_train}))
        
print("r2 :", sess.run(r2, feed_dict={x: x_test, y_: y_test}))
print("loss :", sess.run(cost, feed_dict={x: x_train, y_: y_train}))
sess.close()

# r2 : 1.0
# loss : 1904.5573