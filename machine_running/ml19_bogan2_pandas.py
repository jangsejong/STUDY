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

data = pd.DataFrame([2, np.nan, np.nan, 8, 10], [2, 4, np.nan, 8, np.nan], [np.nan, 4, np.nan, 8, 10], [np.nan, 4, np.nan, 8, np.nan])

data = data.transpose()
data.colums = ["a", "b" ,"c", "d"]

print(data)

#결측치 확인

print(data.isnull().sum())
print(data.info)

#1. 결측치 삭제
# print(data.dropna())
# print(data.dropna(axis=1))
# print(data.dropna(axis=0))


#2-1. 특정값(평균값)
means = data.mean()
print(means)
data = data.fillna(means)

#2-2. 특정값(중위값)
meds = data.median()
data = data.fillna(meds)

#2-3. 특정값(ffill,bfill)
data2 = data.fillna(method="ffill")
print(data2)
data3 = data.fillna(method="bfill")
print(data3)

data2 = data.fillna(method="ffill", limit=1)
print(data2)
data3 = data.fillna(method="bfill", limit=1)
print(data3)

#2-4. 특정값(채우기)
data5 = data.fillna(747474)
print(data5)


# ##############################특정칼럼만

# meds = data['a'].mean()
# print(meds)

# data['a'] = data['a'].fillna(meds)
# print(data)











# from datetime import datetime
# dates = []
# dates = pd.to_datetime(dates)

# ts = pd.Series([2, np.nan, np.nana, 8, 10], index=dates)

# ts = ts.interpolate()
# print(ts)