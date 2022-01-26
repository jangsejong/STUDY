from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine, load_iris, load_boston, load_diabetes
from sklearn.datasets import load_breast_cancer, fetch_covtype,fetch_california_housing

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets= load_boston()
# datasets= load_diabetes()
# datasets = fetch_california_housing()  #link : https://wikidocs.net/49986
#print(datasets.DESCR)
# x = datasets.data
# y = datasets.target
x, y = datasets["data"], datasets["target"]

# y = y.astype(int)
print("LDA 전:", x.shape)    #LDA 전: (442, 10)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66 ,stratify=y)  #shuffle 은 기본값 True
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#차원을 줄여준다.
from sklearn.decomposition import PCA
# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# print(x.shape, y.shape)    #(506, 8) (506,)




#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBRegressor()
# model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)#, eval_metric='error')

#4. 평가, 예측

from sklearn import metrics
y_predict = model.predict(x_train) 
score = metrics.r2_score(y_train, y_predict)
print("r2_score :", score) #1.0
'''

'''
results = model.score(x_test, y_test)
print("결과 : ", results)
'''

'''

# results = model.score(x_test, y_test)
# print("결과 : ", results)

print("LDA 후:", x_train.shape)    #

# import sklearn as sk
# print(sk.__version__)
'''
load_boston
LDA 전: (506, 13)
r2_score : 0.9999989460833698
결과 :  0.8606780455021774
LDA 후: (404, 13)

datasets= load_diabetes()
LDA 전: (442, 10)
r2_score : 0.9999983929235514
결과 :  0.313354229055848
LDA 후: (353, 10)

fetch_california_housing
LDA 전: (20640, 8)
r2_score : 0.8183892476619523
결과 :  0.6624278208677861
LDA 후: (16512, 5)

'''