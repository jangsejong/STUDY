from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias') 

#2. 모델구성

# hypothesis = tf.matmul(x, w) + b
# hypothesis = tf.compat.v1.sigmoid(hypothesis)
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b) # 시그모이드 함수를 사용하면 더 좋은 결과를 낼 수 있다.
# model.add(Dense(1, activation='sigmoid'))

#3. 최적화

# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1 - hypothesis)) # binary_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=4e-2)
train = optimizer.minimize(loss)

#4. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

feed_dicts= {x:x_data, y:y_data}

for epoch in range(70001):
    # sess.run(train, feed_dict=feed_dicts)
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict=feed_dicts)
    if epoch % 1000 == 0:
        print(epoch, "loss:", loss_val, "\n", hy_val)
    

#5. 평가
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 확률값이 0.5 이상이면 1, 아니면 0

#6. 예측

#7. 테스트

#8. 성능평가

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict=feed_dicts)

print("예측결과 :", pred, "\n", "ACC :", acc)
