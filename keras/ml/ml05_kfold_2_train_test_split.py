import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델이다
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


#1. 데이터
datasets= load_wine()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold


# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=66)


# scoring = 'neg_root_mean_squared_error'
scoring = 'cross_val_score'

n_splits =5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)#, scoring=scoring)
'''
cross_val_score는 여러 파라미터를 받습니다.
첫번째는 모델
두번째는 feature
세번째는 target 
cv는 분할 설정값
scoring은 평가방법
score = cross_val_score(dt, iris.data, iris.target, cv=kfold, scoring="accuracy")
print(score.mean())
'''

print("Acc :", scores, "\n cross_val_score :", round(np.mean(scores),4))


'''
#일반적인 k-fold
label이 고르게 분포되지 못하는 현상이 발생합니다.
#shuffle=True
어느정도 고르게 나옵니다.
#StratifiedKFold
더 고르게 분할됩니다.

1) 교차검증의 목적은 모델의 성능 평가를 일반화하는것

2) sklearn의 kFold는 label을 고르게 분배하지 않음

3) sklearn의 cross_val_score 사용해서 kfold 교차검증 수행
'''