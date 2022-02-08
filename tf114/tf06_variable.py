import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
tf.compat.v1.global_variables_initializer().run(session=sess)

print(sess.run(x))

