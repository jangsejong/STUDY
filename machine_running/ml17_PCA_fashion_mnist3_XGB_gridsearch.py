'''
n_components > 0.95이상
XGboost.gridsearch 또는 randomsearch

'''
# Import Libraries
#%matplotlib inline
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
# from sklearn.preprocessing import LabelEncoderfrom 
from tensorflow.keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import datasets
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
np.set_printoptions(threshold=np.inf, linewidth=500)

#1 데이터

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

x = np.append(x_train, x_test, axis=0)

mnist_fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle Boot']

figure = plt.figure(figsize=(15, 10))
for index, i in enumerate(np.random.randint(0, x_train.shape[0], 15)):
    ax = figure.add_subplot(3, 5, index + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap = 'gray')
    ax.set_title(f"{y_train[i]} : {mnist_fashion_labels[y_train[i]]}")
# plt.show()

plt.figure(figsize=(12,12))
plt.imshow(x_train[8703], cmap ='gray')
plt.title(f'{y_train[8703]} : {mnist_fashion_labels[y_train[8703]]}')
plt.xticks([])
plt.yticks([])
# plt.show()

#4.1 머신러닝, 딥러닝을 위한 데이터 처리 - Validation Data Split
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, stratify = y_train, test_size = 0.2, random_state = 87)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


#4.2 머신러닝, 딥러닝을 위한 데이터 처리 - Stratify
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))

X_train.min(),  X_train.max()

#5.1 Machine Learning - 데이터 처리
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=87)
X_train_reshape = X_train.reshape(X_train.shape[0], 28 * 28) # 48000, 784
X_valid_reshape = X_valid.reshape(X_valid.shape[0], 28 * 28) # 12000, 784
x_test_reshape = x_test.reshape(x_test.shape[0], 28 * 28) # 10000, 784

# #5.2 Machine Learning - DecisionTree 학습
# clf = DecisionTreeClassifier(random_state=87)
# cv_score = cross_val_score(clf, X_train_reshape, y_train, cv = skfold)
# # print('\nAccuracy: {:.4f}'.format(cv_score.mean()))

# #5.3 Machine Learning - DecisionTree 검증
# clf.fit(X_train_reshape, y_train)
# # print('\nAccuracy: {:.4f}'.format(clf.score(X_valid_reshape, y_valid)))


# # 5.4 Machine Learning- RandomForest 학습
# rf = RandomForestClassifier(random_state = 87)
# cv_score = cross_val_score(rf, X_train_reshape, y_train, cv = skfold)
# # print('\nAccuracy: {:.4f}'.format(cv_score.mean()))

# # 5.4 Machine Learning - RandomForest 검증
# rf.fit(X_train_reshape, y_train)
# # print('\nAccuracy: {:.4f}'.format(rf.score(X_valid_reshape, y_valid)))

from xgboost import XGBClassifier
# 5.5 Machine Learning- XGBoost 학습
xg = XGBClassifier(random_state = 87)
cv_score = cross_val_score(xg, X_train_reshape, y_train, cv = skfold)
print('\nAccuracy: {:.4f}'.format(cv_score.mean()))

# 5.5 Machine Learning- XGBoost 검증
xg.fit(X_train_reshape, y_train)
print('\nAccuracy: {:.4f}'.format(rf.score(X_valid_reshape, y_valid)))


# #6.1 Deep Learning - 데이터 처리
# y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
# y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, 10)

# #6.2 Deep Learning - Multi Layer Perceptron Model
# mlp_model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape = (28, 28)),
#     tf.keras.layers.Dense(1000, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(800, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(500, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(300, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(200, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(50, activation = 'relu'),
#     tf.keras.layers.Dense(10, activation = 'softmax'),
# ])


# mlp_model.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint = ModelCheckpoint(filepath='model.weights.best.mlp.develop.hdf5', verbose=0, save_best_only=True)
# earlystopping = EarlyStopping(monitor='val_loss', patience=50)
# mlp_history = mlp_model.fit(X_train, y_train_one_hot, epochs=500, batch_size=500,
#                             validation_split=0.2, callbacks=[checkpoint, earlystopping], verbose=0)


# fig, loss_ax = plt.subplots(figsize=(12, 12))
# acc_ax = loss_ax.twinx()

# loss_ax.plot(mlp_history.history['loss'], 'y', label='train loss')
# loss_ax.plot(mlp_history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper right')

# acc_ax.plot(mlp_history.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(mlp_history.history['val_accuracy'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')

# # plt.show()

# #6.7 Deep Learning - Multi Layer Perceptron 예측
# mlp_model.load_weights('model.weights.best.mlp.develop.hdf5')
# print('\nAccuracy: {:.4f}'.format(mlp_model.evaluate(X_valid, y_valid_one_hot)[1]))

# #6.8 Deep Learning - Convolution Neural Network 데이터 전처리
# X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_valid_cnn = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
# X_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

# #6.9 Deep Learning - Convolution Neural Network Model
# cnn_model = tf.keras.Sequential([
#     keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (28, 28, 1)),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size = 2),
#     keras.layers.Dropout(0.3),
    
#     keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size = 2),
#     keras.layers.Dropout(0.3),
    
#     keras.layers.Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size = 2),
#     keras.layers.Dropout(0.3),
    
#     keras.layers.Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size = 2),
#    keras.layers.Dropout(0.3),
    
#     keras.layers.Flatten(), # Flaatten으로 이미지를 일차원으로 바꿔줌
#     keras.layers.Dense(1024, activation = 'relu'),
#     keras.layers.Dense(512, activation = 'relu'),
#     keras.layers.Dense(256, activation = 'relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(10, activation = 'softmax')
# ])

# '''
# 6.9 Deep Learning - Convolution Neural Network Model
# input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다. (행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
# activation : 활성화 함수를 설정
# linear : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
# relu : rectifier 함수, 은익층에 주로 쓰입니다. 0이하는 0으로 만들고 그 이상은 그대로 출력합니다.
# sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. 0 혹은 1로 출력합니다.
# softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다. 0과 1사이의 값으로 출력되며, 모두 더한값은 1이 되므로, 확률처럼 사용합니다.
# filter(kernel) : 이미지의 특징을 찾기위한 파라미터, 해당 filter가 이미지를 움직이며 특징을 잡아냄, 해당 특징이 featuremap, filter의 종류에 따라 가로선 filter, 세로선 filter등이 있는데 cnn에선 해당 필터를 자동으로 생성함
# featuremap : input 이미지에서 filter로 만들어진 해당 이미지의 특성을 가진 map
# filters : input 이미지에서 featuremap을 생성 하는 filter의 갯수
# padding : 외곽의 값을 0으로 채워넣어서 filter들로 만들어진 featuremap 기존의 이미지의 크기와 같게 할지의 여부 same은 같게, valid는 다르게, same으로 하면 filter가 이미지 사이즈에 맞게 featuremap을 만듬.
# pooling : 계속 filter가 이미지를 움직이며 featuremap을 만들고 paddind이 same이라면 계속 같은 크기의 featuremap이 생성되게 됨. 이를 방지하기 위해 pooling 레이어를 거쳐 이미지 사이즈를 줄임, pool_size는 이미지에서 줄여지는 값
# maxpooling : pooling 영역에서 가장 큰 값만 남기는것
# averagepoolig : pooling 영역의 모든 데이터의 평균값을 구하여 남김
# dropout : 이미지의 일부분을 drop시켜 학습하는데 어려움을 줌, 이로 인해 과적함을 막을 수 있습니다.
# flatten : 앞에서 만든 (7, 7, 64)의 배열을 7 * 7 * 64하여 3136의 1줄의 배열로 평평하게 만드는것 입니다.
# (2번의 pooling으로 이미지 사이즈가 작아짐, 28, 28, 64 -> 14, 14, 64 -> 7, 7, 64)
# dense : 평평한 데이터가 들어오면, 해당 데이터를 dense레이어를 지나 맨앞의 사이즈로 줄여줌, 마지막은 10개 사이즈가 나오고, 이를 softmax함수로 활성화하여 0 ~ 9까지의 클래스를 예측할수 있게 해줍니다
# '''

# cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# checkpointer = ModelCheckpoint(filepath='model.weights.best.cnn.hdf5', verbose=0, save_best_only=True)
# earlystopping = EarlyStopping(monitor='val_loss', patience=50)
# history = cnn_model.fit(X_train_cnn, y_train_one_hot, batch_size=500, epochs=500, verbose=0, validation_split=0.2, callbacks=[checkpointer, earlystopping])

# fig, loss_ax = plt.subplots(figsize=(12, 12))
# acc_ax = loss_ax.twinx()

# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper right')

# acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')

# # plt.show()

# cnn_model.load_weights('model.weights.best.cnn.hdf5')
# print('\nAccuracy: {:.4f}'.format(cnn_model.evaluate(X_valid_cnn, y_valid_one_hot)[1]))
 

'''
#XGBoost 비교
F1 = [97,95,90]
Recall = [97,95,90]
Features = [784,100,10]
Time = ['1min 29s','1min 18s','13s']
Models = ['XG Boost (simple_model)', 'XG Boost(using PCA)', 'XG
          Boost(using PCA)']
data = { 'Models': Models,"Features":Features,"Time":Time,'F1' : F1,
         'Recall' : Recall}
dfff = pd.DataFrame(data)
dfff.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})


# t-SNE 기법으로 데이터 시각화
from sklearn.manifold import TSNE
data_1000 = standardized_data[0:1000,:]
labels_1000 = labels[0:1000]
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2",
                 "label"))
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1',
                 'Dim_2').add_legend()
plt.show()


#t-SNE에서 더 많은 매개 변수로 시각화
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=5000)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2",
                 "label"))
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1',
                 'Dim_2').add_legend()
plt.title('With perplexity = 50, n_iter=5000')
plt.show()


#Python으로 KNN 구현
kVals = range(1, 30, 2)
accuracies = []
#We will use for loop to over the various k value for good accuracy
for k in range(1, 30, 2):
 model = KNeighborsClassifier(n_neighbors=k)
 model.fit(x_train, y_train)
 #creating a list of k values and accuracy
 score = model.score(valData, valLabels)
 print("k=%d, accuracy=%.2f%%" % (k, score * 100))
 accuracies.append(score)
 
 이제 k = 5 값으로 모델을 다시 훈련하고 PCA없이 분류 보고서를 가져옵니다.
 
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
from sklearn.metrics import classification_report
print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))
 
 
 이제 400 개의 PCA 구성 요소로 정확도를 찾을 수 있습니다.
 
 
pca = IncrementalPCA(n_components=400)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# test data
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,y_pred))
 
 
 
 
Results
: 비교했을 때, 차원 축소 기술을 사용할 때 축소 과정에서 정보의 일부가 손실 되었기 때문에 정확도가 항상 감소했습니다. 
 

#5.2 Machine Learning - DecisionTree 학습
Accuracy: 0.7900
#5.3 Machine Learning - DecisionTree 검증
Accuracy: 0.7896  


# 5.4 Machine Learning- RandomForest 학습
Accuracy: 0.8799
# 5.5 Machine Learning- RandomForest 검증
Accuracy: 0.8807

#6.7 Deep Learning - Multi Layer Perceptron
Accuracy: 0.8935

cnn_model
Accuracy: 0.9277

  
 '''