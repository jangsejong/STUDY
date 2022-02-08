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
        
# wb = []
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
#     for step in range(2001): # 학습 횟수
#         # sess.run(train) # train 실행
#         _, loss_val, W_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]}) # train 실행
#         if step % 100 == 0: # 학습 횟수마다 출력
#             # print(step, sess.run(loss), sess.run(w), sess.run(b)) # 각 반복마다 출력   
#             print(step, loss_val, W_val, b_val) # 각 반복마다 출력
#             wb.append(sess.run([w, b]))
#     #4. Test
#     print("예측값:", sess.run(hypothesis, feed_dict={x_train:[4]})) # [3.995577]
#     print("예측값:", sess.run(hypothesis, feed_dict={x_train:[5,6]})) # 예측값: [4.993016 5.990454]
#     print("예측값:", sess.run(hypothesis, feed_dict={x_train:[6,7,8]})) #예측값: [5.990454  6.9878926 7.985331 ]
    
    

# #5. 결과 확인
# sess = tf.InteractiveSession() # 자동으로 default session을 지정해줌 
# sess.run(tf.global_variables_initializer())
# c = sess.run([4])
# print(c.eval()) # 제대로 작동 
# print(sess.run(c)) 
# sess.close()

    
