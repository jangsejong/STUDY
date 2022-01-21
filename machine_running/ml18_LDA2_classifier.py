from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine, load_iris, load_boston, load_diabetes
from sklearn.datasets import load_breast_cancer, fetch_covtype

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets= load_iris()
# datasets= load_breast_cancer()
# datasets= load_wine()
# datasets= fetch_covtype()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
print("LDA 전:", x.shape)    #LDA 전: (150, 4)/////
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66 ,stratify=y)  #shuffle 은 기본값 True


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()


#차원을 줄여준다.
from sklearn.decomposition import PCA
# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()
# x = pca.fit_transform(x)
x = lda.fit_transform(x, y)

print(x.shape, y.shape)    #(506, 8) (506,)




#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

print("LDA 후:", x.shape)    #LDA 후: (150, 2)/////

# import sklearn as sk
# print(sk.__version__)
'''
결과 :  0.9824561403508771

datasets= load_iris()
결과 :  0.9666666666666667

datasets= load_breast_cancer()
결과 :  0.956140350877193

datasets= load_wine()
결과 :  0.9722222222222222

datasets= fetch_covtype()
결과 :  0.8695128352968512
'''