from pickletools import optimize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense , Dropout, Conv2D, Flatten, Input
import time
import numpy as np


#1 데이터

(x_train, y_train),(x_test,y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# n = x_train.shape[0]
# x_train = x_train.reshape(n,-1)/255.
x_train = x_train.reshape(60000,28*28).astype('float32')/255

# m = x_test.shape[0]
# x_test = x_test.reshape(m,-1)/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델

def build_model(drop = 0.5, optimizer = 'adam', activation = 'relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs' )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss= 'categorical_crossentropy')
    
    return model




def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizer = ['adam','rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid']
    return {"batch_size" : batchs, "optimizer" : optimizer,
            "drop": dropout, "activation" : activation}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# 텐서플로를 사이킷런 형태로 래핑해준다. 

kerasclassifier = KerasClassifier(build_fn= build_model, verbose=1)



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = GridSearchCV(kerasclassifier, hyperparameters, cv=3)

import time
start = time.time()
model.fit(x_train, y_train, verbose=1, epochs = 30, validation_split=0.2)#, callbacks=[es,mcp], batch_size =1000)
end = time.time()

model.save("./_save/keras2_1_save_model.h5")
# model = load_model("./_save/keras25_save_model.h5")

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_: ", model.best_score_)
print("model.score_: ", model.score)


# print("model.score: ", model.score(x_test, y_test)) #score는 evaluate 개념


from sklearn.metrics import accuracy_score
y_predict=model.predict(x_test)

print("걸린시간 :", end - start)
print("accuracy_score: ", accuracy_score(y_test, y_predict))

#가중치 세이브

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 acc :", accuracy_score(y_test,y_pred_best))
