#  실습
# [4]
# [5,6]
# [6,7,8]
# 위 값들을 프레딕 해라
# x_test 라는 placeholder 생성







import numpy as np 
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. Data
# x_train = [1,2,3] # x_train = np.array([1,2,3])
# y_train = [1,2,3] # y_train = np.array([1,2,3])
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
x_test = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(1 , name='weight', dtype= tf.float32) #초기값을 1로 설정
# b = tf.Variable(1 , name='bias', dtype= tf.float32) #초기값을 1로 설정
w = tf.Variable(tf.random_normal([1]), name='weight', dtype = tf.float32) #초기값을 1로 설정
b = tf.Variable(tf.random_normal([1]), name='bias', dtype = tf.float32) #초기값을 1로 설정

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화

# print(sess.run(w))  #[0.06524777]



# 2. Model

hypothesis = x_train * w + b # y = xw + b

#3-1. compile

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법
train = optimizer.minimize(loss) # 최적화 함수
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

#3-2. run

with tf.compat.v1.Session() as sess:        # tf.~~~Session()을 sess로써 실행하고 작업이 다 끝나면 종료해라.
# sess.close()    # session은 항상 열었으면 닫아주어야한다. with문을 쓰면 자동으로 종료된다.
# sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    step = 0
    while True:
        step += 1
        # sess.run(train)     # 여기서 실행이 일어난다.
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
        
        if step % 20 == 0:
            # print(f"{step+1}, {sess.run(loss)}, {sess.run(w)}, {sess.run(b)}")
            print(step, loss_val, w_val, b_val)
        
        if w_val >= 0.99:
            
            predict = x_test*w+b
            
            predict = sess.run(predict,feed_dict={x_test:[6,7,8]})
            print(predict)
            break
        

    
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predict = x_test * w_val + b_val # y_predict = model.predict(x_test)

# print("[6,7,8]의 예측값:", sess.run(y_predict, feed_dict={x_test:[6,7,8]})) # [5.990454  6.9878926 7.985331 ]