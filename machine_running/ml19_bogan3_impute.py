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
from xgboost import sklearn

'''
#결측치 처리
행 또는 열 삭제
임의의 값
fillna - 0, ffill, bfill, 중위값, 평균값,,, 76767
보간 - Interpolate
모델링 - predict
부스팅계열 - 통상 결측치, 이상치에 대해 자유롭다. 믿거나 말거나
'''

data = pd.DataFrame([2, np.NaN, np.NaN, 8, 10], [2, 4, np.NaN, 8, np.NaN], [np.NaN, 4, np.NaN, 8, 10], [np.NaN, 4, np.nan, 8, np.NaN])

data = data.transpose()
data.colums = ["a", "b" ,"c", "d"]

print(data)

#결측치 확인

# print(data.isnull().sum())
# print(data.info)

# meds = data['a'].mean()
# print(meds)

# data['a'] = data['a'].fillna(meds)
# print(data)


# median = data['b'].median()
# print(meds)

# data['b'] = data['b'].fillna(median)
# print(data)

# from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# # imputer = SimpleImputer(strategy='most_frequent')
# # imputer = SimpleImputer(strategy='constant')
# # imputer = SimpleImputer(strategy='constant', fill_value=777)
# '''
# strategy 옵션
#     'mean': 평균값 (디폴트)
#     'median': 중앙값
#     'most_frequent': 최빈값
#     'constant': 특정값, 예 SimpleImputer(strategy='constant', fill_value=777)
# '''

# imputer.fit(data)
# data2 = imputer.transform(data)

# print(data2)

# # ##############################특정칼럼만

# # meds = data['a'].mean()
# # print(meds)

# # data['a'] = data['a'].fillna(meds)
# # print(data)

