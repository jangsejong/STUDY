import tensorflow as tf
print(tf.__version__)

print(tf.executing_eagerly()) #True

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly()) #False

hello= tf.constant("helllo world")

sess = tf.compat.v1.Session()

print(sess.run(hello))


