import numpy as np 
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. Data
# x_train = [1,2,3] # x_train = np.array([1,2,3])
# y_train = [1,2,3] # y_train = np.array([1,2,3])
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(1 , name='weight', dtype= tf.float32) #초기값을 1로 설정
# b = tf.Variable(1 , name='bias', dtype= tf.float32) #초기값을 1로 설정
w = tf.Variable(tf.random_normal([1]), name='weight', dtype = tf.float32) #초기값을 1로 설정
b = tf.Variable(tf.random_normal([1]), name='bias', dtype = tf.float32) #초기값을 1로 설정

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화

print(sess.run(w))  #[0.06524777]



# 2. Model

hypothesis = x_train * w + b # y = xw + b

#3-1. compile

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법
train = optimizer.minimize(loss) # 최적화 함수
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

#3-2. run

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
#     for step in range(2001): # 학습 횟수
#         # sess.run(train) # train 실행
#         _, loss_val, W_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]}) # train 실행
#         if step % 100 == 0: # 학습 횟수마다 출력
#             # print(step, sess.run(loss), sess.run(w), sess.run(b)) # 각 반복마다 출력   
#             print(step, loss_val, W_val, b_val) # 각 반복마다 출력

sess = tf.compat.v1.Session() # 세션 생성
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
for step in range(2001): # 학습 횟수
    # sess.run(train) # train 실행
    _, loss_val, W_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]}) # train 실행
    
    if step % 100 == 0: # 학습 횟수마다 출력
        print(step, sess.run(loss), sess.run(w), sess.run(b)) # 각 반복마다 출력

#4. Evaluate
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)

print("[6,7,8]의 예측값:", sess.run(y_predict, feed_dict={x_test:[6,7,8]})) # [5.990454  6.9878926 7.985331 ]


sess.close() # 세션 닫기        

'''
import tensorflow as tf
 
x_data = [1, 2, 3]
y_data = [1, 2, 3]
 
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
 
hypothesis = X * W
 
cost = tf.reduce_mean(tf.square(hypothesis - Y))
 
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
 
sess = tf.Session()
 
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

'''



