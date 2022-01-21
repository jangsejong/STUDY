from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper


#1 데이터

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)

print(x_train.shape, x_test.shape)    #(60000, 28, 28) (10000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


from sklearn.decomposition import PCA
pca = PCA(n_components=486)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)

# print(cumsum)

print(np.argmax(cumsum >=0.95)+1)  #154
print(np.argmax(cumsum >=0.99)+1)  #331
print(np.argmax(cumsum >=0.999)+1)  #486
print(np.argmax(cumsum)+1)  #713   #  1.0





from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import time



# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

n = x_train.shape[0]
x_train = x_train.reshape(n,-1)/255.

m = x_test.shape[0]
x_test = x_test.reshape(m,-1)/255.

# scaler =MinMaxScaler()   
# x_train = scaler.fit_transform(x_train)     
# x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# model.add(Dense(128, input_shape = (28*28,)))
model.add(Dense(256, input_dim = 784))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 30
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k34_dnn_mnist_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size =1000)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("acc : ",loss[1])

'''
<DNN 결과>>
시간 :  14.58 초

acc : 0.9788333177566528

<CNN 결과> >>>>>>>>>>>>>>>>>>>>
loss :  0.1922997087240219
acc :  0.9574000239372253

시간 :  14.58 초


# 0.95 n_components > 17
간 :  15.99 초
loss :  0.19689007103443146
acc :  0.9538999795913696

# 0.99  n_components > 21
시간 :  17.76 초
loss :  0.17632824182510376
acc :  0.9563999772071838

# 1.0 n_components 
loss :  0.12691718339920044
acc :  0.9675999879837036

'''



