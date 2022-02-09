from turtle import update
from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(77)


# 1. 데이터

x_train_data = [1, 2, 3]
y_train_data = [1, 2, 3]
x_test_data = [4, 5, 6]
y_test_data = [4, 5, 6]


x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
x_test = tf.compat.v1.placeholder(tf.float32)
y_test = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random_normal([1]) , name='weight') #초기값을 랜덤으로 주어서 실험을 하는 것이 좋다.


#2. 모델구성
hypothesis = x * w       

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

lr = 0.2
gradient = tf.reduce_mean((w * x - y) * x )
descent = w - lr * gradient
update = w.assign(descent) # w = w - lr * gradient

sess = tf.compat.v1.Session() # 세션 생성
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화

w_history = []
loss_history = [] 
feed_dicts = {x: x_train_data, y: y_train_data} 

for step in range(21):
    # sess.run(update, feed_dict=feed_dicts) # w = w - lr * gradient
    # print(step, sess.run(loss, feed_dict=feed_dicts), sess.run(w))
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict=feed_dicts)
    # print(step, loss_v, w_v)
    
    # w_history.append(w_v) # w_v 값을 저장
    # loss_history.append(loss_v) # loss_v 값을 저장
    
# sess.close()

# predict = x_test_data * w_v # 테스트 데이터를 이용해서 예측값을 구한다.
# predict_data = sess.run(predict, feed_dict={x_test: x_test_data})
# # print(predict)
# print(predict_data)

# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# r2_score(predict, sess.run(w) * x_test_data)
# print("R2 : ", r2_score(predict, sess.run(w) * x_test_data))

y_predict = x_test_data * w_v # 테스트 데이터를 이용해서 예측값을 구한다.
# y_predict_data = sess.run(y_predict, feed_dict={x_test: x_test_data})
print(y_predict)
# print(y_predict_data)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_predict, y_test_data)
print("R2 : ", r2)

mae = mean_absolute_error(y_predict, y_test_data)
print("MAE : ", mae)

sess.close()
