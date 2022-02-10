from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(66)

sess = tf.compat.v1.Session()
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
print(sess.run(tf.cast(x, tf.int32)))