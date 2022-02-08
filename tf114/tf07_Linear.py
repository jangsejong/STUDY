import numpy as np 
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. Data
x_train = [1,2,3] # x_train = np.array([1,2,3])
y_train = [1,2,3] # y_train = np.array([1,2,3])

w = tf.Variable(1 , name='weight', dtype= tf.float32) #초기값을 1로 설정
b = tf.Variable(1 , name='bias', dtype= tf.float32) #초기값을 1로 설정


#2. Model

hypothesis = x_train * w + b # y = xw + b

#3-1. compile

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1) # 경사하강법
'''
learning_rate = 0.1 # 학습률
gradient = tf.reduce_mean((W * X - Y) * X) # d/dW
descent = W - learning_rate * gradient #경사하강법
update = W.assign(descent) # 업데이트
'''
train = optimizer.minimize(loss) # 최적화 함수


#3-2. run

sess = tf.compat.v1.Session() # 세션 생성
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화

for step in range(5): # 학습 횟수
    sess.run(train) # train 실행
    if step % 1 == 0: # 학습 횟수마다 출력
        print(step, sess.run(loss), sess.run(w), sess.run(b)) # 각 반복마다 출력


