import numpy as np #pd는 np로 구성되어있다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
import tensorflow as tf

#1. 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

#print(x.shape, y.shape)  #(4, 3) (4,)

#input_shape = (batch_size, timesteps, feature) /  행,렬,자르는갯수

x = x.reshape(4, 3, 1)
#y = y.reshape(4, 1)
#x = np.array(x, dtype=np.float32)
print(y.shape)




#2.모델구성
model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(3, 1)))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(4, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
# activation = tanh


# Long-Short-Term Memory
: LSTM도 똑같이 체인과 같은 구조를 가지고 있지만, 각 반복 모듈은 다른 구조를 갖고 있다. 단순한 neural network layer 한 층 대신에, 4개의 layer가 특별한 방식으로 서로 정보를 주고 받도록 되어 있다.
STM은 cell state에 뭔가를 더하거나 없앨 수 있는 능력이 있는데, 이 능력은 gate라고 불리는 구조에 의해서 조심스럽게 제어된다.
Gate는 정보가 전달될 수 있는 추가적인 방법으로, sigmoid layer와 pointwise 곱셈으로 이루어져 있다.
Sigmoid layer는 0과 1 사이의 숫자를 내보내는데, 이 값은 각 컴포넌트가 얼마나 정보를 전달해야 하는지에 대한 척도를 나타낸다. 그 값이 0이라면 "아무 것도 넘기지 말라"가 되고, 값이 1이라면 "모든 것을 넘겨드려라"가 된다.
LSTM은 3개의 gate를 가지고 있고, 이 문들은 cell state를 보호하고 제어한다.
LSTM도 똑같이 체인과 같은 구조를 가지고 있지만, 각 반복 모듈은 다른 구조를 갖고 있다. 단순한 neural network layer 한 층 대신에, 4개의 layer가 특별한 방식으로 서로 정보를 주고 받도록 되어 있다.


link : https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr


===============================================

순전파(Feedforward)와 역전파(Backpropagation) 개념
: 다층 퍼셉트론(Multi-layer Perceptron, MLP)으로 학습 한다는 것은 최종 출력값과 실제값의 오차가 최소화 되도록 
  가중치와 바이어스를 계산하여 결정하는 것이다. 순전파 (Feedforward) 알고리즘 에서 발생한 오차를 줄이기 위해 새로운 가중치를 업데이트하고, 
  새로운 가중치로 다시 학습하는 과정을 역전파 (Backpropagation) 알고리즘 이라고 한다. 이러한 역전파 학습을 오차가0에 가까워 질 때까지 반복한다. 
  역전파 알고리즘을 실행할때 가중치를 결정하는 방법에서는 경사하강법이 사용된다.

[순전파 (Feedfoward)]

입력층에서 은닉층 방향으로 이동하면서 각 입력에 해당하는 가중치가 곱해지고, 결과적으로 가중치 합으로 계산되어 은닉층 뉴런의 함수 값(일반적으로 시그모이드(Sigmoid) 사용)이 입력된다. 
그리고 최종 결과가 출력된다

[역전파 (Backpropagation)]

역전파 알고리즘은 input과 output 값을 알고 있는 상태에서 신경망을 학습 시키는 방법이다. 
이 방법을 Supervised learning(지도학습)이라고 한다. 초기 가중치, weight 값은 랜덤으로 주어지고 각각 노드들은 하나의 퍼셉트론으로, 노드를 지날때 마다 활성함수를 적용한다.








'''



