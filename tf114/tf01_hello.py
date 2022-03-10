import tensorflow as tf
print(tf.__version__)

# print("hello")

hello1= tf.constant("helllo world")
# hello3= tf.placeholder("helllo world")
print(hello1)  
# print(hello3)  

# constant 상수를 정의하는 것이 아니라 값을 정의하는 것이다.
# variable 변수를 정의하는 것이 아니라 값을 정의하는 것이다.
# 이런 차이를 이용해서 상수와 변수를 정의하는 것을 이해하는 것이 중요하다.
# 상수는 값을 정의하고 변수는 값을 정의하는 것이다.
# placeholder는 실제 값을 넣어주는 것이 아니라 값을 정의하는 것이다.
# 이런 차이를 이용해서 실제 값을 넣어주는 것을 이해하는 것이 중요하다.



sess = tf.Session()
print(sess.run(hello1))









