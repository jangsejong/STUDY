import warnings, gc
from math import ceil, sqrt
from decimal import ROUND_HALF_UP, Decimal
from datetime import datetime, timedelta
import pickle

import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import interpolate

import matplotlib.colors
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, AveragePooling1D, GlobalAveragePooling1D
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

import jpx_tokyo_market_prediction

warnings.filterwarnings("ignore")

pd.options.display.max_rows = 50
pd.options.display.min_rows = 50
pd.options.display.max_columns = None

tqdm.pandas()

# Test config
test_config = {
    #Specify which step to run to reduce testing time
    'load_data': True,
    'calc_features': True,
    'train_cv': False,
    'predict_cv': False,
    'train_sector': True,
    'predict_sector': True,
    
    # Specify if we use XGBoost with GPU or not
    'use_gpu': True,
    'early_stopping_rounds': 5,
    'verbose': 25,
    'ignore_count': 0
}

# XGBoost training parameters
xgb_submit_params = {
    'verbosity': 1,
    'objective': 'reg:squarederror',
    'n_estimators': 10_000,
    'learning_rate': 0.02,
    'max_depth': 14,
    'random_state': 21,
    'tree_method': 'hist'
}

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

y_train_nodate = train_data.drop('ymdhm', axis = 1)
y_valid_nodate = train_label.drop('ymdhm', axis = 1)


# 추론 데이터셋 만들기

test_data = []
test_label = []

tmp = pd.read_csv(w_list[-1])
tmp = tmp.replace(" ", np.nan)

print(train_data.shape)
print(train_label.shape)



# 모델링 및 모델 학습

input_shape = (train_data[0].shape[0], train_data[0].shape[1])

# Training

model = XGBRegressor(**xgb_submit_params).fit(train_data, train_label, 
                                              eval_set=[(test_data, test_label)], 
                                              verbose = test_config['verbose'], 
                                              early_stopping_rounds = test_config['early_stopping_rounds'])

# optimizer = tf.optimizers.RMSprop(0.001)

# model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])




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
