from pickletools import optimize
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
      
x_data = [[73., 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]

y_data = [[152], [185], [180], [196], [142]] #(5, 1)



x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([3,1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias') 

#2. 모델구성

hypothesis = tf.matmul(x, w) + b

#3. 최적화

loss = tf.reduce_mean(tf.square(hypothesis - y))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=4e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=4e-5)

train = optimizer.minimize(loss)

#4. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

feed_dicts= {x:x_data, y:y_data}

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
R2 :  0.9996362453297368
MSE :  0.153067965246737
AdamOptimizer
R2 :  0.9997270494853739
MSE :  0.11485757655464113
'''







