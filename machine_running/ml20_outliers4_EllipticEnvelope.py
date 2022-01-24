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


aaa = np.array([[1,2,-1000,4,5,6,7,8,90,100,500,12,13],[100,200,3,400,500,600,7,800,900,190,1001,1002,99]])
#(2, 13) -> (13, 2)

aaa = np.transpose(aaa)


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) # 10%구간 오염도로 잡는다.

outliers.fit(aaa)
resulits = outliers.predict(aaa)
print(resulits)



