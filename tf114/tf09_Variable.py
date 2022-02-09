import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 
w = tf.Variable(tf.random_normal([1]), name='weight', dtype = tf.float32) #초기값을 1로 설정
print(w)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
aaa = sess.run(w) 
print("aaa:", aaa) # [0.06524777]
sess.close()

#2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
bbb = w.eval(session=sess) # w.eval()
print("bbb:", bbb) # [0.06524777]
sess.close()

#3. 
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
ccc = w.eval() # w.eval()
print("ccc:", ccc) # [0.06524777]
sess.close()





