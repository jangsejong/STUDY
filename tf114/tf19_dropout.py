import tensorflow as tf


x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([2,8]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([8]), name = 'bias1')



#2. 모델구성

hedden_layer1 = tf.sigmoid(tf.matmul(x , w1) + b1)
# hedden_layer1 = tf.matmul(x , w1) + b1
# hedden_layer1 = tf.nn.relu(tf.sigmoid(tf.matmul(x , w1) + b1))

w2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([8,1]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name = 'bias2')

hidden_layer1 = tf.sigmoid(tf.matmul(hedden_layer1, w2) + b2)
layers = tf.nn.dropout(hidden_layer1, keep_prob = 0.7)