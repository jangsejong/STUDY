from unittest import result
import tensorflow as tf
tf.set_random_seed(66)

x_data =[[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,6,7]]

y_data =[[0,0,1],
         [0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

x_predict = [[1,11,7,9]] # 결과값을 예측하기 위해 입력값을 설정한다.


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random.normal([4,3]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name='bias') 

#2. 모델구성

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) #
# model.add = Dense(3, activation='softmax')

#3. 최적화

# loss = tf.reduce_mean(tf.square(hypothesis - y))
# loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1 - hypothesis)) # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=4e-2)
train = optimizer.minimize(loss)

#4. 훈련

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

feed_dicts= {x:x_data, y:y_data}

# for epoch in range(70001):
#     sess.run(train, feed_dict=feed_dicts)
#     if epoch % 2000 == 0:
#         print(epoch, sess.run(loss, feed_dict=feed_dicts))
    

#5. 평가
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 확률값이 0.5 이상이면 1, 아니면 0

#6. 예측

#7. 테스트

#8. 성능평가

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32)) # 예측값과 실제값이 같으면 1, 아니면 0
# accuracy = tf.reduce_mean(tf.cast(y!=y_predict, dtype=tf.float32)) # link : https://velog.io/@jangsejong/%EA%B3%BC-equals-%EC%9D%98-%EC%B0%A8%EC%9D%B4%EC%A0%90

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
    # sess.run(tf.global_variables_initializer())
    for step in range(70001):
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dicts)
        if step % 2000 == 0:
            print(step, loss_val)
    #predict
    results = sess.run(hypothesis, feed_dict={x:x_predict})
    print(results, sess.run(tf.arg_max(result, 1)))

pred, acc = sess.run([y_predict, accuracy], feed_dict=feed_dicts)

print("예측결과 :", pred, "\n", "ACC :", acc)




loss = sess.run(loss, feed_dict=feed_dicts)
print('loss : ', loss)
# loss :  0.025958158
# accuracy :  1.0


    
