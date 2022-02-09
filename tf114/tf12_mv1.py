from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
        # 첫변 둘변 세변  네변  다섯변
x1_data = [73., 93., 89., 96., 73.] # 국어
x2_data = [80., 88., 91., 98., 66.] # 영어
x3_data = [75., 93., 90., 100., 70.] #수학
y_data = [152., 185., 180., 196., 142.] #환산점수

# x는 (5,3), y는 (5,1) 또는 (5,)
# y = w1x1 + w2x2 + w3x3 + b

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias') 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run([w1, w2, w3]))

#2. 모델구성

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

#3. 최적화
loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=4e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

#4. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

feed_dicts = {x1:x1_data, x2:x2_data, x3:x3_data, y:y_data}

for epochs in range(70001):
    sess.run(train, feed_dict=feed_dicts)
    if epochs % 1000 == 0:
        print(epochs, sess.run(loss, feed_dict=feed_dicts))
    


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

'''
GradientDescentOptimizer
R2 :  0.9995678548904502
MSE :  0.18184666209854186

AdamOptimizer
R2 :  0.9996663950516014
MSE :  0.1403809622861445



'''