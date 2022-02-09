import tensorflow as tf
tf.set_random_seed(77)
import matplotlib.pyplot as plt

# 1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]
w = tf.placeholder(tf.float32, name='weight')


#2. 모델구성
hypothesis = x * w       

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w: curr_w})
        # print("{} * {} = {}".format(i, w, sess.run(hypothesis, feed_dict={w: i})))
        # print("mse : ", curr_loss)
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("========================================================")
print("w_history : ", w_history)
print("========================================================")
print("loss_history : ", loss_history)

import matplotlib
from matplotlib import font_manager, rc
import platform
#matplotlib 에서 사용하는 폰트를 한글 지원이 가능한 것으로 바꾸는 코드
if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

matplotlib.rcParams['axes.unicode_minus'] = False   
#그래프에서 마이너스 기호가 표시되도록 하는 설정입니다. 

plt.plot(w_history, loss_history)
plt.xlabel('웨이트')
plt.ylabel('로스')
plt.title('선생님 만세')
plt.show()