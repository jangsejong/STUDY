import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 

#1. 데이터
x = np.array([range(1,2,3,4,5)])
y = np.array([range(1,2,3,4,5)])

model = Sequential()
model.add(Dense(3, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print(model.weights)
# print(model.get_weights()) # 가중치들이 모두 곱해진 값을 반환
print(model.trainable_weights) # 가중치를 가지고 있는 모델의 가중치를 리스트로 반환

model.trainable = False # 가중치를 가지고 있는 모델의 가중치를 모두 무효화
