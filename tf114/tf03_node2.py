import tensorflow as tf

node1  = tf.constant(2.0, tf.float32)
node2 = tf.constant(3.0)
node3 = node1 + node2
node3 = tf.add_n([node1, node2])  # add /  add_n 많은양의 텐서를 한번에 처리
node4 = node1 - node2
node4 = tf.subtract(node1, node2)
node5 = node1 * node2
node5 = tf.multiply(node1, node2)  # multiply 원소곱 matmul  행렬곱
node6 = node1 / node2
node6 = tf.divide(node1, node2)   
node7 = tf.math.mod(node1, node2)   # mod 나눈 나머지




print(node3)
# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))
print(sess.run(node7))











