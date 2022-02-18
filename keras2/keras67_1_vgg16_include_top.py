import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.keras.applications import VGG16 # 기존 모델을 사용하기 위해 임포트


model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 


model.summary()



'''
include_top=True : 훈련된 모델의 가중치를 사용하여 새로운 모델을 생성하는 것이 아니라, 기존 모델의 가중치를 사용하여 새로운 모델을 생성한다.

 fc1 (Dense)                 (None, 4096)              102764544

 fc2 (Dense)                 (None, 4096)              16781312

 predictions (Dense)         (None, 1000)              4097000

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

=================================================================
include_top=False : 기존 모델의 가중치를 사용하지 않고 새로운 모델을 생성한다.



=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0


많은 층의 레이어를 갖게 되면서 레이어들을 구별하기 위해, 연구자들은 각 레이어마다 이름을 붙여줍니다. 
컨볼루션 레이어라면 conv1, conv2, conv3와 같이, fully-connected 레이어라면 fc1, fc2, fc3와 같이 말이죠. 
그런데 만약 우리가 tensorflow.keras를 이용해서 딥러닝 모델을 만들면, 각 레이어의 이름이 임의로 생성됩니다. 

Fully connected input layer (평탄화)
━이전 레이어의 출력을 "평탄화"하여 다음 스테이지의 입력이 될 수 있는 단일 벡터로 변환합니다.

The first fully connected layer
━이미지 분석의 입력을 취하여 정확한 라벨을 예측하기 위해 가중치를 적용합니다.

Fully connected output layer
━각 라벨에 대한 최종 확률을 제공합니다.


Fully connected layer의 목적은 Convolution/Pooling 프로세스의 결과를 취하여 이미지를 정의된 라벨로 분류하는 데 사용하는 것입니다(단순한 분류의 예).

FC(Fully connected layer)를 정의하자면,

완전히 연결 되었다라는 뜻으로,

한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로

2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층입니다.

1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화
2. 활성화 함수(Relu, Leaky Relu, Tanh,등)뉴런을 활성화
3. 분류기(Softmax) 함수로 분류

1~3과정을 Fully Connected Layer라고 말할 수 있습니다.

'''