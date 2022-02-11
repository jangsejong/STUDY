import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() #mnist.load_data()
train_images = train_images.reshape([-1, 784]) # 28 * 28 = 784
test_images  = test_images.reshape([-1,784]) # 28*28
train_images = train_images / 255. # 정규화
test_images  = test_images / 255. # 정규화
print(train_images[0]) # [0.  0.  0. ..., 0.  0.  0.]
#train_labels = train_labels.reshape([-1, 784]) # 이렇게 하면 에러남
print('train_images.shape : ', train_images.shape) # (60000, 784)
#from tensorflow.examples.tutorials.mnist import input_data

tf.disable_v2_behavior() # 에러 방지

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # one_hot=True 인코딩



nb_classes = 10 # 0 ~ 9까지 10개

X = tf.compat.v1.placeholder(tf.float32, [None, 784]) # 입력값
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes]) # 출력값

W = tf.compat.v1.Variable(tf.random_normal([784, nb_classes])) # 가중치
b = tf.compat.v1.Variable(tf.random_normal([nb_classes])) # 편향

#batch_xs, batch_ys = mnist.train.next_batch(100) # 100개씩 잘라서 배치

hypothesis = tf.compat.v1.nn.softmax(tf.matmul(X, W) + b) # 예측값
cost = tf.compat.v1.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1)) # 손실함수
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost) # 최적화 함수

prediction = tf.compat.v1.argmax(hypothesis, 1) #예측한 결과를 0~6사이의 값으로 만든다 # 예측값이 가장 큰 숫자의 인덱스를 리턴
is_correct = tf.compat.v1.equal(prediction, tf.argmax(Y, 1))#예측한 결과와 Y 데이터를 비교 # 예측값과 실제값이 같은지 비교
accuracy = tf.compat.v1.reduce_mean(tf.cast(is_correct, tf.float32)) #이것들을 평균낸다 # 정확도

training_epochs = 1000 # 학습 횟수
batch_size = 100 # 배치 크기

import matplotlib.pyplot as plt
import random

#'''
with tf.Session() as sess: # 세션 생성
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for epoch in range(training_epochs): # 학습 횟수만큼 반복
        avg_cost = 0
        total_batch = int(train_images.shape[0] / batch_size) # 전체 데이터수를 배치 크기로 나눈다

        for i in range(total_batch):
            s_idx = int(train_images.shape[0] * i / total_batch) # 시작 인덱스
            e_idx = int(train_images.shape[0] * (i+1)/ total_batch) # 끝 인덱스
            #print('s_idx : ', s_idx)
            #print('width : ', width)
            batch_xs = train_images[s_idx : e_idx] # 배치 이미지
            batch_ys = train_labels[s_idx : e_idx] # 배치 라벨
            #print('batch_xs.shape : ', batch_xs.shape) # (100, 784)
            #print('batch_ys.shape : ', batch_ys.shape) # (100, 10)
            #Y_one_hot = tf.one_hot(batch_ys, nb_classes) # 원핫인코딩
            Y_one_hot = np.eye(nb_classes)[batch_ys] # one_hot 인코딩
            #print('Y_one_hot.shape :', Y_one_hot.shape) # (100, 10)
            _,c = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:Y_one_hot}) # 학습
            #print('total_batch : ', total_batch, ', c:', c) # 손실함수 값
            avg_cost += c / total_batch # 손실함수 평균값

        print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost)) # 현재 횟수와 현재 손실함수 값 출력
    Y_one_hot = np.eye(nb_classes)[test_labels] # one_hot 인코딩
    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X:test_images, Y:Y_one_hot})) # 정확도 출력
    
    r = random.randint(0, test_images.shape[0] - 1) # 랜덤 인덱스
    print('label : ', test_labels[r:r+1]) # 실제 라벨
    print('Prediction : ', sess.run(tf.argmax(hypothesis, 1), feed_dict={X:test_images[r:r+1]})) # 예측값
    plt.imshow(test_images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest') # 이미지 출력
    plt.show() # 이미지 출력
#'''
'''
Accuracy :  0.9255
label :  [5]
Prediction :  [5]
'''