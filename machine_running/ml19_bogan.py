from operator import index
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
#결측치 처리
행 또는 열 삭제
임의의 값
fillna - 0, ffill, bfill, 중위값, 평균값,,, 76767
보간 - Interpolate
모델링 - predict
부스팅계열 - 통상 결측치, 이상치에 대해 자유롭다. 믿거나 말거나
'''

from datetime import datetime
dates = []
dates = pd.to_datetime(dates)

ts = pd.Series([2, np.nan, np.nana, 8, 10], index=dates)

ts = ts.interpolate()
print(ts)