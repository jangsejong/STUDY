from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
##model.add(Conv2D(10, (2, 2), padding='valid', input_shape=(10, 10, 1), activation='relu')) # (9, 9, 10)
model.add(Conv2D(10 ,kernel_size=(2,2), input_shape=(10, 10, 1)))                          # (9, 9, 10)
model.add(Conv2D(5,kernel_size=(3,3), activation='relu'))                                  # (7, 7, 5)
model.add(Dropout(0.2))
model.add(Conv2D(7,kernel_size=(2,2), activation='relu'))                                  # (6, 6, 7)
model.add(Flatten())                                                                       # (None, 252) 
model.add(Dense(30, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(18, activation='linear'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='linear'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='softmax'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147
=================================================================
Total params: 652
Trainable params: 652
Non-trainable params: 0
_________________________________________________________________

param  (2*2*10)*1+10     (3*3*5)*10+5  (2*2*7)*5+7
       (커널행*커널열*채널수*현노드수)*상단노드수+바이어스  

첫번째 인자 : 컨볼루션 필터의 수 입니다. 
두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
padding : 경계 처리 방법을 정의합니다.
‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
activation : 활성화 함수 설정합니다.
‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

output_shape : ?
'''



