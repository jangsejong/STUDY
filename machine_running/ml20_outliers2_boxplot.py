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


aaa = np.array([1,2,-10,4,5,6,7,8])#,90,100,500,12,13])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :", quartile_1)
    print("q2 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 :", outliers_loc)



import matplotlib.pyplot as plt

import seaborn as sns

# Basic box plot
# plt.boxplot(aaa)
# plt.show()


# setting outlier symbol, title, xlabel
plt.boxplot([aaa], sym="bo")
plt.title('Box plot of tip')
plt.xticks([1], ['aaa'])
plt.show()


# # 컬럼 기준으로 값이 하나라도 비어있는 컬럼의 이름을 모은다

# cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# # 모은 컬럼 버리기(drop)

# reduced_X_train = X_train.drop(cols_with_missing, axis=1)









