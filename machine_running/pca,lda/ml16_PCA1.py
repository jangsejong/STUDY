from codecs import ignore_errors
from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
import sklearn as sk

#1. 데이터
# datasets = load_boston()
# datasets = fetch_california_housing()
datasets = load_breast_cancer()


x = datasets.data
y = datasets.target
print(x.shape)    # (506, 13)
# print(x.shape)    # (20640, 8)

pca = PCA(n_components=8)
x = pca.fit_transform(x)
print(x)
print(x.shape)  # (506, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

print(sk.__version__)