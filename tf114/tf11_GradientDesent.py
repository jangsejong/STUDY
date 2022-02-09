from turtle import update
from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(77)


# 1. 데이터

x_train = [1, 2, 3]
y_train = [1, 2, 3]



x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

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
feed_dicts = {x: x_train, y: y_train} 

for step in range(21):
    sess.run(update, feed_dict=feed_dicts) # w = w - lr * gradient
    print(step, sess.run(loss, feed_dict=feed_dicts), sess.run(w))
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict=feed_dicts)
    print(step, loss_v, w_v)
    
    w_history.append(w_v) # w_v 값을 저장
    loss_history.append(loss_v) # loss_v 값을 저장
    
# sess.close()

print("========================================================")
print("w_history : ", w_history)
print("========================================================")
print("loss_history : ", loss_history)
print("w : ", sess.run(w))

sess.close()
print(f"{step:04d}\t{update_val:.5f} \t{descent_val:.5f} \t{gradient_val:.5f} \t{loss_val:.5f} \t{w_val:.5f}")
# plt.plot(w_history, loss_history)
# plt.xlabel('w')
# plt.ylabel('loss')
# plt.show()


'''

'''