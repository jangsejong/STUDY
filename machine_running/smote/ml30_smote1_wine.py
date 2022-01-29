from sympy import im
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score, r2_score, accuracy_score
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE


#1.데이터 로드
# datasets = fetch_california_housing()
# datasets = load_boston()
datasets = load_wine()
# datasets = fetch_covtype()

# path = 'D:\\Study\\_data\\dacon\\whitewine\\'
# # datasets = pd.read_csv(path +"winequality-white.csv", thousands=",", encoding='cp949',sep=';')
# datasets = pd.read_csv(path + "winequality-white.csv",delimiter=';')




x = datasets.data
y = datasets.target

print(np.unique(y, return_counts=True))
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48

x_new = x[ :-30]
y_new = y[ :-30]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split (x_new, y_new, shuffle=True, random_state=66, train_size=0.75, stratify=y_new)
'''
매개변수
arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)
train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
shuffle : 셔플여부설정 (default = True)
stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할됩니다.
(알아보기 위해선, np.bincount로, 테스트와 학습 데이터를 확인하면 됩니다.)
'''

print("============SMOTE 적용==============")
smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

from xgboost import XGBClassifier

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

model_score = model.score(x_test, y_test) 
print("results :", round(model_score,4))
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", round(acc, 4))