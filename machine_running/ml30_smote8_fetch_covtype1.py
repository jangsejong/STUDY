import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, load_boston, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import *
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier

#1. 데이터
import pickle
x_train = pickle.load(open('./_save/m30_x_train_save.dat', 'rb'))
x_test = pickle.load(open('./_save/m30_x_test_save.dat', 'rb'))
y_train = pickle.load(open('./_save/m30_y_train_save.dat', 'rb'))
y_test = pickle.load(open('./_save/m30_y_test_save.dat', 'rb'))

