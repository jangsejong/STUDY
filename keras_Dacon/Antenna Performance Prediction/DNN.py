#구글 드라이브 연결
from google.colab import drive

drive.mount('/content/gdrive/')

#구글 드라이브 경로 설정 
DATA_PATH = '/content/gdrive/MyDrive/Dacon/LG_Aimers/data/' 
MODEL_PATH='/content/gdrive/MyDrive/Dacon/LG_Aimers/model/'
SUBMISSION_PATH='/content/gdrive/MyDrive/Dacon/LG_Aimers/submission/' 

import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import xgboost as xgb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(15) # Seed 고정

# Data Load

train_df = pd.read_csv(DATA_PATH + 'train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue : 56
train_y = train_df.filter(regex='Y') # Output : Y Feature : 14

# {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': 3, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 1000, 'power_t': 0.5, 'solver': 'adam', 'warm_start': False}

#learning_rate = 0.01 에서 0.0003 으로 바꿈

# MLPRegressor

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['LeakyReLU', 'relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.0003],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, verbose=True, pre_dispatch='2*n_jobs')
mlp_GS.fit(train_x, train_y)

print(mlp_GS.best_params_)

test_x = pd.read_csv(DATA_PATH + 'test.csv').drop(columns=['ID'])

preds = mlp_GS.predict(test_x)
print('Done.')

submit = pd.read_csv(DATA_PATH +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv(SUBMISSION_PATH + 'mlp.csv', index=False)
