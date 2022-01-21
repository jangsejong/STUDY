from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
datasets= load_breast_cancer()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(178, 13) (178,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

#차원을 줄여준다.
from sklearn.decomposition import PCA
# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()
# x = pca.fit_transform(x)
x = lda.fit_transform(x, y)

print(x.shape, y.shape)    #(506, 8) (506,)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  #shuffle 은 기본값 True

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


# import sklearn as sk
# print(sk.__version__)
'''
결과 :  0.9824561403508771
'''