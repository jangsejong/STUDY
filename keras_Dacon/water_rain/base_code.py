import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import interpolate

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, AveragePooling1D, GlobalAveragePooling1D

#GPU 자원이 부족한 경우 아래 코드를 이용하세요
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

w_list = sorted(glob("D:\study\Dacon\competition_data\팔당댐\water_data\*.csv"))
w_list

# pd.read_csv(w_list[0]).shape
# pd.read_csv(w_list[0]).head(4)

train_data = []
train_label = []
num = 0

for i in w_list[:-1]:
    
    tmp = pd.read_csv(i)
    tmp = tmp.replace(" ", np.nan)
    tmp = tmp.interpolate(method = 'values')
    tmp = tmp.fillna(0)
    
    for j in tqdm(range(len(tmp)-432)):
        train_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                       "tototf", "tide_level",
                                                       "fw_1018662", "fw_1018680",
                                                       "fw_1018683", "fw_1019630"]]).astype(float))
        
        train_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                      "wl_1018683", "wl_1019630"]]).astype(float))


train_data = np.array(train_data)
train_label = np.array(train_label)

print(train_data.shape)
print(train_label.shape)


# 모델링 및 모델 학습

input_shape = (train_data[0].shape[0], train_data[0].shape[1])

model = Sequential()
model.add(GRU(256, input_shape=input_shape))
model.add(Dense(4, activation = 'relu'))

optimizer = tf.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])

model.fit(train_data, train_label, epochs=10, batch_size=128)


# 추론 데이터셋 만들기

test_data = []
test_label = []

tmp = pd.read_csv(w_list[-1])
tmp = tmp.replace(" ", np.nan)
# 이전값을 사용
tmp = tmp.fillna(method = 'pad')
tmp = tmp.fillna(0)
    
#tmp.loc[:, ["wl_1018662", "wl_1018680", "wl_1018683", "wl_1019630"]] = tmp.loc[:, ["wl_1018662", "wl_1018680", "wl_1018683", "wl_1019630"]]*100
    
for j in tqdm(range(4032, len(tmp)-432)):
    test_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                    "tototf", "tide_level",
                                                    "fw_1018662", "fw_1018680",
                                                    "fw_1018683", "fw_1019630"]]).astype(float))
        
    test_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                    "wl_1018683", "wl_1019630"]]).astype(float))

test_data = np.array(test_data)
test_label = np.array(test_label)

print(test_data.shape)
print(test_label.shape)

# 제출 파일 만들기

pred = model.predict(test_data)

pred = pd.DataFrame(pred)

sample_submission = pd.read_csv("D:\study\Dacon\competition_data\팔당댐\sample_submission.csv")

sample_submission["wl_1018662"] = pred[0]
sample_submission["wl_1018680"] = pred[1]
sample_submission["wl_1018683"] = pred[2]
sample_submission["wl_1019630"] = pred[3]

sample_submission.to_csv("baseline.csv", index = False)

