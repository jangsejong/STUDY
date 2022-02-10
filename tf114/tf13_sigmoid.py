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
    sess.run(train, feed_dict=feed_dicts)
    if epoch % 1000 == 0:
        print(epoch, sess.run(loss, feed_dict=feed_dicts))
    

#5. 평가
predict = sess.run(hypothesis, feed_dict=feed_dicts) 

#6. 예측

#7. 테스트

#8. 성능평가

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_data, predict)
print("R2 : ", r2)

mse = mean_squared_error(y_data, predict)
print("MSE : ", mse)

sess.close()

'''
R2 :  0.8837209038883364
MSE :  0.029069774027915923

시그모이드 함수를 사용
R2 :  0.9904949148924321
MSE :  0.0023762712768919743

binary_crossentropy 함수를 사용
R2 :  0.9995551708547561
MSE :  0.00011120728631097377

'''