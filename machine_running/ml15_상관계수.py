from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

#1. 데이터
datasets= load_iris()
# print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(178, 13) (178,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

# df = pd.DataFrame(x, columns=datasets['feature_names'])
# df = pd.DataFrame(x, columns=datasets.feature_names)

df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])

df['Target(y)'] = y

print(df)
print("-----------------상관계수 하드 맵------------------")
print(df.corr())

import matplotlib.pyplot
import seaborn
sns.ser(font_scalw=1.2)
sns.heatmap(data=df.corr())

